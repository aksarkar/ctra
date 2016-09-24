"""Fit the hierarchical model

We fit a generalized linear model regressing phenotype against genotype. We
impose a spike-and-slab prior on the coefficients to regularize the
problem. Our inference task is to estimate the posterior distribution of the
parameters pi (probability each SNP is causal) and tau (precision of causal
effects).

The inference requires integrating an intractable posterior over the
hyperparameters. Our strategy is to use importance sampling to perform the
integration, where the importance weights are the model evidence. We estimate
the importance weights by fitting a variational approximation to the
intractable posterior p(theta, z | x, y).

We cannot write an analytical solution for the variational approximation, so we
take a doubly stochastic approach, using Monte Carlo integration to estimate
intractable expectations (re-parameterizing integrals as sums) and drawing
samples (due to the non-conjugate prior) to estimate the gradient. We
additionally use a control variate to reduce the variance of the gradient
estimator.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import itertools
import logging
import sys

import numpy
import scipy.misc
import scipy.special
import theano
import theano.tensor as T

import ctra.pcgc

logger = logging.getLogger(__name__)

_real = theano.config.floatX
_F = theano.function
_S = lambda x: theano.shared(x, borrow=True)
_Z = lambda n: numpy.zeros(n).astype(_real)
_R = numpy.random
_N = lambda n: _R.normal(size=n).astype(_real)

# We need to change base since we're going to be interpreting the value of pi,
# logit(pi)
_expit = lambda x: scipy.special.expit(x * numpy.log(10))
_logit = lambda x: scipy.special.logit(x) / numpy.log(10)

class Model:
    """Class providing the implementation of the optimizer

    This is intended to provide a pickle-able object to re-use the Theano
    compiled function across hyperparameter samples.

    """
    def __init__(self, X_, y_, a_, minibatch_n=None, stoch_samples=None,
                 learning_rate=None, *params, **kwargs):
        """Compile the Theano function which takes a gradient step"""
        logger.debug('Building the Theano graph')
        # Observed data
        n, p = X_.shape
        X = _S(X_.astype(_real))
        y = _S(y_.astype(_real))
        a = _S(a_.astype('int8'))
        self.var_x = X_.var(axis=0).sum()

        # Hyperparameters
        pi = T.vector(name='pi')
        pi_deref = T.basic.choose(a, pi)
        tau = T.vector(name='tau')
        tau_deref = T.basic.choose(a, tau)

        # Variational parameters
        alpha_raw = _S(_Z(p))
        eps = numpy.finfo(_real).eps
        alpha = T.cast(T.clip(T.nnet.sigmoid(alpha_raw), eps, 1 - eps), _real)
        beta = _S(_N(p))
        gamma_raw = _S(_Z(p))
        gamma = 1e5 * T.nnet.sigmoid(gamma_raw)

        self.params = [alpha_raw, beta, gamma_raw]
        self.params.extend(params)

        # We need to perform inference on minibatches of samples for speed. Rather
        # than taking balanced subsamples, we take a sliding window over a
        # permutation which is balanced in expectation.
        epoch = T.iscalar(name='epoch')
        if minibatch_n is not None:
            perm = _S(_R.permutation(n).astype('int32'))
            sample_minibatch = epoch % (n // minibatch_n)
            index = perm[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]
            X_s = X[index]
            y_s = y[index]
        else:
            minibatch_n = n
            X_s = X
            y_s = y

        # Variational approximation (re-parameterize eta = X theta). This is a
        # "Gaussian reconstruction" in that we characterize its expectation and
        # variance, then approximate its distribution with a Gaussian.
        #
        # We need to take the gradient of an intractable integral, so we re-write
        # it as a Monte Carlo integral which is differentiable, following Kingma &
        # Welling, ICLR 2014 (http://arxiv.org/abs/1312.6114).
        mu = T.dot(X_s, alpha * beta)
        nu = T.dot(T.sqr(X_s), alpha / gamma + alpha * (1 - alpha) * T.sqr(beta))
        random = T.shared_randomstreams.RandomStreams(seed=0)
        if stoch_samples is None and minibatch_n > 10:
            stoch_samples = 1
        else:
            stoch_samples = 10
        eta_raw = random.normal(size=(stoch_samples, minibatch_n))
        eta = mu + T.sqrt(nu) * eta_raw

        # Objective function
        elbo = (
            # The log likelihood is for the minibatch, but we need to scale up
            # to the full dataset size
            self._llik(y_s, eta) * (n // minibatch_n)
            + .5 * T.sum(alpha * (1 + T.log(tau_deref) - T.log(gamma) - tau_deref * (T.sqr(beta) + 1 / gamma)))
            - T.sum(alpha * T.log(alpha / pi_deref) + (1 - alpha) * T.log((1 - alpha) / (1 - pi_deref)))
        )
        self._elbo = elbo

        eta_mean = T.addbroadcast(T.mean(eta, keepdims=True), (True))
        control = [self._llik(y_s, eta_mean) * g
                   for g in T.grad(T.mean(-T.sqr(eta_mean - mu) / T.sqrt(nu)),
                                   self.params)]

        logger.debug('Compiling the Theano functions')
        self._randomize = _F(inputs=[], outputs=[],
                             updates=[(alpha_raw, _Z(p)), (beta, _N(p))])
        a = T.vector()
        b = T.vector()
        self._initialize = _F(inputs=[a, b], outputs=[],
                              updates=[(alpha_raw, a), (beta, b)],
                              allow_input_downcast=True)

        grad = T.grad(elbo, self.params)
        if learning_rate is None:
            learning_rate = numpy.array(5e-2 / minibatch_n, dtype=_real)
        logger.debug('Minibatch size = {}'.format(minibatch_n))
        logger.debug('Initial learning rate = {}'.format(learning_rate))
        self.vb_step = _F(inputs=[epoch, pi, tau],
                          outputs=elbo,
                          updates=[(p_, T.cast(p_ + 10 ** -(epoch // 1e5) * learning_rate * (g - cv), _real))
                                   for p_, g, cv in zip(self.params, grad, control)])

        self._opt = _F(inputs=[pi, tau], outputs=[alpha, beta])
        logger.debug('Finished initializing')

    def _llik(self, *args):
        raise NotImplementedError

    def _log_weight(self, params=None, weight=0.5, poll_iters=1000,
                    min_iters=100000, atol=1, **hyperparams):
        """Return optimum ELBO and variational parameters which achieve it.

        params - Initial setting of the variational parameters (default: randomize)
        weight - weight for exponential moving average of ELBO
        poll_iters - number of iterations before polling objective function
        min_iters - minimum number of iterations
        atol - maximum change in objective for convergence
        hyperparams - pi, tau, etc.

        """
        logger.debug('Starting SGD given {}'.format(hyperparams))
        # Re-initialize, otherwise everything breaks
        if params is None:
            self._randomize()
        else:
            self._initialize(scipy.special.expit(alpha), beta)
        converged = False
        t = 0
        ewma = 0
        ewma_ = ewma
        while not converged:
            t += 1
            elbo = self.vb_step(epoch=t, **hyperparams)
            assert numpy.isfinite(elbo)
            if t < poll_iters:
                ewma += elbo / poll_iters
            else:
                ewma *= (1 - weight)
                ewma += weight * elbo
            if not t % poll_iters:
                logger.debug('Iteration = {}, EWMA = {}'.format(t, ewma))
                if ewma_ < 0 and (ewma < ewma_ or numpy.isclose(ewma, ewma_, atol=atol)):
                    converged = True
                else:
                    ewma_ = ewma
                    alpha, beta = self._opt(**hyperparams)
        return ewma, (alpha, beta)

    def fit(self, **kwargs):
        """Return the posterior mean estimates of the hyperparameters

        kwargs - algorithm-specific parameters

        """
        # Propose pi_0, pi_1. This induces a proposal for tau_0, tau_1
        # following Guan et al. Ann Appl Stat 2011; Carbonetto et al. Bayesian
        # Anal 2012
        proposals = list(itertools.product(*[numpy.arange(-3, 0, 0.5)
                                             for a in self.pve]))

        # Find the best initialization of the variational parameters. In
        # general we need to warm start different sets of parameters per model,
        # so in this generic implementation we can't unpack things.
        params = None
        pi = numpy.zeros(shape=(len(proposals), self.pve.shape[0]), dtype=_real)
        tau = numpy.zeros(shape=pi.shape, dtype=_real)
        best_elbo = float('-inf')
        logger.info('Finding best initialization')
        for i, logit_pi in enumerate(proposals):
            pi[i] = _expit(numpy.array(logit_pi)).astype(_real)
            tau[i] = (((1 - self.pve) * pi[i] * self.var_x) / self.pve).astype(_real)
            elbo_, params_ = self._log_weight(pi=pi[i], tau=tau[i], **kwargs)
            if elbo_ > best_elbo:
                params = params_

        # Perform importance sampling, using ELBO instead of the marginal
        # likelihood
        log_weights = numpy.zeros(shape=len(proposals))
        logger.info('Performing importance sampling')
        for i, logit_pi in enumerate(proposals):
            log_weights[i], *_ = self._log_weight(pi=pi[i], tau=tau[i],
                                                  params=params, **kwargs)

        # Scale the log importance weights before normalizing to avoid numerical
        # problems
        log_weights -= max(log_weights)
        normalized_weights = numpy.exp(log_weights - scipy.misc.logsumexp(log_weights))
        logger.debug('Importance Weights = {}'.format(normalized_weights))
        self.pi = normalized_weights.dot(pi)
        return self

class LogisticModel(Model):
    def __init__(self, X, y, a, K, pve, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        _bias = theano.shared(numpy.array([0], dtype=_real))
        self.bias = T.addbroadcast(_bias, 0)
        super().__init__(X, y, a, params=[self.bias], **kwargs)
        # Now add terms to ELBO for the bias: q(bias) ~ N(bias; theta_0,
        # 2.5^2), q(theta_0) ~ N(0, 2.5^2)
        self._elbo += self.bias / 2.5
        self.pve = pve
        logger.info('Fixing parameters {}'.format({'pve': self.pve}))

    def _llik(self, y, eta):
        """Return E_q[ln p(y | eta, theta_0)] assuming a logit link."""
        F = y * (eta + self.bias) - T.nnet.softplus(eta + self.bias)
        return T.mean(T.sum(F, axis=1))

class ProbitModel(Model):
    def __init__(self, X, y, a, K, **kwargs):
        super().__init__(X, y, a, **kwargs)
        self.pve = ctra.pcgc.estimate(y, ctra.pcgc.grm(X, a), K)

    def _cdf(eta):
        return .5 * (1 + T.erf(eta / T.sqrt(2)))

    def _llik(self, y, eta):
        """Return E_q[ln p(y | eta)] assuming a logit link."""
        F = y * T.log(_cdf(eta)) + (1 - y) * T.log(1 - _cdf(eta))
        return T.mean(T.sum(F, axis=1))

class GaussianModel(Model):
    def __init__(self, X, y, a, pve):
        self.X = X
        self.y = y
        self.a = a
        self.var_x = X.var(axis=0).sum()
        self.pve = pve
        logger.info('Fixing parameters {}'.format({'pve': self.pve}))

    def _log_weight(self, pi, tau, params=None, atol=1e-4, **hyperparams):
        """Implement the coordinate ascent algorithm of Carbonetto and Stephens,
    Bayesian Anal (2012)

        Generalizing this algorithm to the multi-annotation case is trivial since
        the local updates just refer to the shared hyperparameter.

        """
        X = self.X
        y = self.y
        a = self.a
        n, p = X.shape
        y = y.reshape(-1, 1)
        pi_deref = numpy.choose(a, pi)
        logit_pi = _logit(pi_deref)
        tau_deref = numpy.choose(a, tau)
        # Initial configuration
        if params is None:
            alpha = _R.uniform(size=p)
            alpha /= alpha.sum()
            beta = _R.normal(size=p)
            sigma2 = y.var()
        else:
            alpha, beta, sigma2 = params
        logger.debug('Starting coordinate ascent given {}'.format({'pve': self.pve, 'pi': pi, 'tau': tau, 'sigma2': sigma2, 'var_x': self.var_x}))
        # Precompute things
        eps = numpy.finfo(float).eps
        xty = X.T.dot(y)
        xtx_jj = numpy.einsum('ij,ji->i', X.T, X)
        gamma = (xtx_jj + tau_deref) / sigma2
        eta = X.dot(alpha * beta).reshape(-1, 1)
        # Coordinate ascent
        L = numpy.log
        elbo = float('-inf')
        converged = False
        reverse = False
        logger.debug('{:>8s} {:>8s} {:>8s}'.format('ELBO', 'sigma2', 'Vars'))
        while not converged:
            alpha_ = alpha.copy()
            beta_ = beta.copy()
            elbo_ = elbo
            for j in range(p):
                xj = X[:, j].reshape(-1, 1)
                theta_j = alpha[j] * beta[j]
                beta[j] = (xty[j] + xtx_jj[j] * theta_j - (xj * eta).sum()) / (gamma[j] * sigma2)
                ssr = beta[j] * beta[j] * gamma[j]
                alpha[j] = numpy.clip(scipy.special.expit(numpy.log(10) * logit_pi[j] + .5 * (ssr + L(tau_deref[j]) - L(sigma2) - L(gamma[j]))), eps, 1 - eps)
                eta += xj * (alpha[j] * beta[j] - theta_j)
            sigma2 = (numpy.square(y - eta).sum() +
                      (xtx_jj * alpha * (1 / gamma + (1 - alpha) * beta ** 2)).sum() +
                      (alpha.dot(tau_deref * (1 / gamma + numpy.square(beta))))) / (n + alpha.sum())
            gamma = (xtx_jj + tau_deref) / sigma2
            elbo = (-.5 * (n * L(2 * numpy.pi * sigma2) +
                            numpy.square(y - eta).sum() / sigma2 +
                            (xtx_jj * alpha * (1 / gamma + (1 - alpha) * beta ** 2)).sum() / sigma2) +
                     .5 * (alpha * (1 + L(tau_deref) - L(gamma) - L(sigma2) -
                                    tau_deref / sigma2 * (numpy.square(beta) + 1 / gamma))).sum() -
                     (alpha * L(alpha / pi_deref) +
                      (1 - alpha) * L((1 - alpha) / (1 - pi_deref))).sum())
            logger.debug('{:>8.6g} {:>8.3g} {:>8.0f}'.format(elbo, sigma2, alpha.sum()))
            if elbo > 0:
                raise ValueError('ELBO must be non-positive')
            if elbo < elbo_:
                converged = True
            elif numpy.isclose(alpha_, alpha, atol=atol).all() and numpy.isclose(alpha_ * beta_, alpha * beta, atol=atol).all():
                converged = True
        return elbo, (alpha, beta, sigma2)

class LogitModel(Model):
    def __init__(self, X, y, a, pve):
        self.X = X
        self.y = y
        self.a = a
        self.var_x = X.var(axis=0).sum()
        self.pve = pve
        self.eps = numpy.finfo(float).eps
        logger.info('Fixing parameters {}'.format({'pve': self.pve}))
    
    def _update_logistic_var_params(self, zeta):
        # Variational lower bound to the logistic likelihood (Jaakola & Jordan,
        # 2000)
        d = (scipy.special.expit(zeta) - .5) / (zeta + self.eps)
        bias = (self.y - .5).sum() / d.sum()
        yhat = self.y - .5 - bias * d
        xty = self.X.T.dot(yhat)
        xd = self.X.T.dot(d)
        xdx = numpy.einsum('ij,ik,kj->j', self.X, numpy.diag(d), self.X) - numpy.square(xd) / d.sum()
        return d, yhat, xty, xd, xdx

    def _log_weight(self, pi, tau, params=None, atol=1e-4, **hyperparams):
        """Implement the coordinate ascent algorithm of Carbonetto and Stephens,
    Bayesian Anal (2012)

        Generalizing this algorithm to the multi-annotation case is trivial since
        the local updates just refer to the shared hyperparameter.

        """
        X = self.X
        y = self.y
        a = self.a
        n, p = X.shape
        y = y.reshape(-1, 1)
        pi_deref = numpy.choose(a, pi)
        logit_pi = _logit(pi_deref)
        tau_deref = numpy.choose(a, tau)
        # Initial configuration
        if params is None:
            alpha = _R.uniform(size=p)
            alpha /= alpha.sum()
            beta = _R.normal(size=p)
            zeta = numpy.ones(n)
        else:
            alpha, beta, zeta = params
        logger.debug('Starting coordinate ascent given {}'.format({'pve': self.pve, 'pi': pi, 'tau': tau}))
        # Precompute things
        eta = X.dot(alpha * beta).reshape(-1, 1)
        d, yhat, xty, xd, xdx = self._update_logistic_var_params(zeta)
        gamma = xdx + tau_deref
        # Coordinate ascent
        L = numpy.log
        S = numpy.square
        elbo = float('-inf')
        converged = False
        reverse = False
        logger.debug('{:>8s} {:>8s}'.format('ELBO', 'Vars'))
        while not converged:
            alpha_ = alpha.copy()
            beta_ = beta.copy()
            zeta_ = zeta.copy()
            elbo_ = elbo
            for j in range(p):
                xj = X[:, j].reshape(-1, 1)
                theta_j = alpha[j] * beta[j]
                beta[j] = (xty[j] + xdx[j] * theta_j + xd[j] * d.T.dot(eta) / d.sum() - xj.T.dot(numpy.diag(d)).dot(eta)) / gamma[j]
                ssr = beta[j] * beta[j] * gamma[j]
                alpha[j] = scipy.special.expit(numpy.log(10) * logit_pi[j] + .5 * (ssr + L(tau_deref[j]) - L(gamma[j])))
                eta += xj * (alpha[j] * beta[j] - theta_j)
            gamma = xdx + tau_deref
            a = 1 / d.sum()
            # M step for free parameters zeta
            bias = a * ((y - 0.5).sum() - d.dot(eta))
            # V[theta_j] under the variational approximation
            theta_var = alpha * (1 / gamma + (1 - alpha) * beta ** 2)
            bias_var = a * (1 + a * (theta_var * S(xd)).sum())
            theta_bias_covar = -a * xd * theta_var
            zeta = numpy.sqrt(S(bias + eta).ravel() + bias_var +
                              numpy.einsum('ij,j,ji->i', X, theta_var, X.T) +
                              2 * X.dot(theta_bias_covar))
            d, yhat, xty, xd, xdx = self._update_logistic_var_params(zeta)
            elbo = (L(a) / 2 + a * S((y - 0.5).sum()) / 2 + L(scipy.special.expit(zeta)).sum() +
                    numpy.square(d.dot(eta)) / (2 * d.sum()) +
                    zeta.dot(d * zeta - 1) / 2 + yhat.T.dot(eta) - numpy.einsum('ji,jk,ki', eta, numpy.diag(d), eta) +
                    -.5 * xdx.dot(theta_var) +
                     .5 * (alpha * (1 + L(tau_deref) - L(gamma) - tau_deref * (S(beta) + 1 / gamma))).sum() -
                     (alpha * L(alpha / pi_deref) + (1 - alpha) * L((1 - alpha) / (1 - pi_deref))).sum())
            logger.debug('{:>8.6g} {:>8.0f}'.format(elbo[0], alpha.sum()))
            if not numpy.isfinite(elbo):
                raise ValueError('ELBO must be finite')
            elif elbo > 0:
                raise ValueError('ELBO must be non-positive')
            if elbo < elbo_:
                converged = True
            elif numpy.isclose(alpha_, alpha, atol=atol).all() and numpy.isclose(alpha_ * beta_, alpha * beta, atol=atol).all():
                converged = True
        return elbo, (alpha, beta, zeta)
