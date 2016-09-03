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
samples (due to the non-conjugate prior) to estimate the gradient.

In our stochastic optimization, we use the sample mean of of the individual
sample likelihoods across the random samples eta as a control variate, since
its expectation is 0.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import drmaa
import itertools
import pickle
import os
import sys

import numpy
import numpy.random as R
import scipy.misc
import scipy.special
import scipy.stats
import theano
import theano.tensor as T

import ctra.pcgc

numpy.seterr(all='warn')
_real = theano.config.floatX
_F = theano.function
_S = lambda x: theano.shared(x, borrow=True)
_Z = lambda n: numpy.zeros(n).astype(_real)
# We need to change base since we're going to be interpreting the value of pi,
# logit(pi)
_expit = lambda x: scipy.special.expit(x * numpy.log(10))
_logit = lambda x: scipy.special.logit(x) / numpy.log(10)

class Model:
    """Class providing the implementation of the optimizer

    This is intended to provide a pickle-able object to re-use the Theano
    compiled function across hyperparameter samples.

    """
    def __init__(self, X_, y_, a_, minibatch_n=100, max_precision=1e5,
                 learning_rate=None, b1=0.9, b2=0.999, e=1e-8, **kwargs):
        """Compile the Theano function which takes a gradient step

        llik - data likelihood under the variational approximation
        max_precision - maximum value of gamma
        learning_rate - initial gradient ascent step size (used for Adam)
        b1 - first moment exponential decay (Adam)
        b2 - second moment exponential decay (Adam)
        e - tolerance (Adam)

        """
        print('Building the Theano graph')
        # Observed data
        X = _S(X_.astype(_real))
        y = _S(y_.astype(_real))
        a = _S(a_.astype('int8'))
        n, p = X_.shape
        self.p = p
        self.m = 1 + max(a_)
        self.var_x = X_.var(axis=0).sum()

        # Variational parameters
        alpha_raw = _S(_Z(p))
        alpha = T.nnet.sigmoid(alpha_raw)
        beta = _S(_Z(p))
        gamma_raw = _S(_Z(p))
        gamma = max_precision * T.nnet.sigmoid(gamma_raw)
        self.alpha_raw = alpha_raw
        self.alpha = alpha
        params = [alpha_raw, beta, gamma_raw]

        # Hyperparameters
        pi = T.vector(name='pi')
        pi_deref = T.basic.choose(a, pi)
        tau = T.vector(name='tau')
        tau_deref = T.basic.choose(a, tau)

        # We need to perform inference on minibatches of samples for speed. Rather
        # than taking balanced subsamples, we take a sliding window over a
        # permutation which is balanced in expectation.
        perm = _S(R.permutation(n).astype('int32'))
        epoch = T.iscalar(name='epoch')
        sample_minibatch = epoch % (n // minibatch_n)
        index = perm[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]
        X_s = X[index]
        y_s = y[index]

        # Variational approximation (re-parameterize eta = X theta). This is a
        # "Gaussian reconstruction" in that we characterize its expectation and
        # variance, then approximate its distribution with a Gaussian.
        #
        # We need to take the gradient of an intractable integral, so we re-write
        # it as a Monte Carlo integral which is differentiable, following Kingma &
        # Welling, ICLR 2014 (http://arxiv.org/abs/1312.6114). When we take the
        # gradient, the global mean of the reconstruction is constant and drops
        # out, so we only need to keep the global variance.
        mu = T.dot(X_s, alpha * beta)
        nu = T.dot(T.sqr(X_s), alpha / gamma + alpha * (1 - alpha) + T.sqr(beta))
        random = T.shared_randomstreams.RandomStreams(seed=0)
        eta_raw = random.normal(size=(10, minibatch_n))
        eta = mu + T.sqrt(nu) * eta_raw

        # Objective function
        elbo = (
            self._llik(y_s, eta)
            + .5 * T.sum(alpha * (1 + T.log(tau_deref) - T.log(gamma) - tau_deref * (T.sqr(beta) + 1 / gamma)))
            - T.sum(alpha * T.log(alpha / pi_deref) + (1 - alpha) * T.log((1 - alpha) / (1 - pi_deref)))
        )

        # Maximize ELBO using stochastic gradient ascent
        #
        # Adaptive estimation (Kingma & Welling arxiv:1412.6980) tunes the learning
        # rate based on exponentially weighted moving averages of the first and
        # second moments of the gradient.
        if learning_rate is None:
            learning_rate = 0.5 / n
        grad = T.grad(elbo, params)
        M = [_S(_Z(p)) for param in params]
        V = [_S(_Z(p)) for param in params]
        a_t = learning_rate * T.sqrt(1 - T.pow(b2, epoch)) / (1 - T.pow(b1, epoch))
        adam_updates = collections.OrderedDict()
        for p, g, m, v in zip(params, grad, M, V):
            new_m = b1 * m + (1 - b1) * g
            new_v = b2 * v + (1 - b2) * T.sqr(g)
            adam_updates[p] = T.cast(p + a_t * new_m / (T.sqrt(new_v) + e), _real)
            adam_updates[m] = new_m
            adam_updates[v] = new_v
        print('Compiling the Theano function')
        self.vb_step = _F(inputs=[epoch, pi, tau], outputs=elbo,
                          updates=adam_updates)

    def _llik(self, *args):
        raise NotImplementedError

    def _log_weight(self, **hyperparams):
        """Return optimum ELBO and variational parameters which achieve it

        Use Adaptive estimation to tune the learning rate and converge when
        change in ELBO changes sign. Poll ELBO every thousand iterations to
        reduce sensitivity to stochasticity.

        """
        delta = 1
        print('Starting SGD with hyperparameters = {}'.format(hyperparams))
        # Re-initialize alpha, otherwise everything breaks
        self.alpha_raw.set_value(_Z(self.p))
        converged = False
        t = 1
        curr_elbo = self.vb_step(epoch=t, **hyperparams)
        print('ELBO={:.3g}\tVars={:.0g}'.format(curr_elbo, self.alpha.sum().eval()))
        while delta > 0:
            t += 1
            new_elbo = self.vb_step(epoch=t, **hyperparams)
            if not t % 1000:
                delta = new_elbo - curr_elbo
                if delta > 0:
                    curr_elbo = new_elbo
                    print('ELBO={:.3g}\tVars={:.0g}'.format(curr_elbo, self.alpha.sum().eval()))
        print('ELBO={:.3g}\tVars={:.0g}'.format(curr_elbo, self.alpha.sum().eval()))
        return curr_elbo, alpha, beta

    def fit(self):
        """Return the posterior mean estimates of the hyperparameters"""
        # Propose pi_0, pi_1. This induces a proposal for tau_0, tau_1
        # following Guan et al. Ann Appl Stat 2011; Carbonetto et al. Bayesian
        # Anal 2012
        logit_pi_proposal = numpy.linspace(-5, 0, 10)
        proposals = logit_pi_proposal
        m = len(proposals)

        # Find the best initialization
        alpha = None
        beta = None
        pi = numpy.zeros(shape=(m, 1), dtype=_real)
        tau = numpy.zeros(shape=(m, 1), dtype=_real)
        best_elbo = float('-inf')
        print('Finding best initialization')
        for i, logit_pi in enumerate(proposals):
            pi[i] = _expit(numpy.array(logit_pi)).astype(_real)
            tau[i] = (((1 - self.pve) * pi[i] * self.var_x) / self.pve).astype(_real)
            elbo_, alpha_, beta_= self._log_weight(pi=pi[i], tau=tau[i])
            if elbo_ > best_elbo:
                alpha, beta = alpha_, beta_

        # Perform importance sampling, using ELBO instead of the marginal
        # likelihood
        log_weights = numpy.zeros(shape=m)
        print('Performing importance sampling')
        for i, logit_pi in enumerate(proposals):
            log_weights[i], *_ = self._log_weight(pi=pi[i], tau=tau[i], alpha=alpha, beta=beta)

        # Scale the log importance weights before normalizing to avoid numerical
        # problems
        log_weights -= max(log_weights)
        normalized_weights = numpy.exp(log_weights - scipy.misc.logsumexp(log_weights))
        print('Weights =', normalized_weights)
        self.pi = normalized_weights.dot(pi)
        return self

    def predict(self, x):
        return _expit(x.dot(self.alpha * self.beta))

class LogisticModel(Model):
    def __init__(self, X, y, a, K, **kwargs):
        super().__init__(X, y, a, **kwargs)
        self.pve = ctra.pcgc.estimate(y, ctra.pcgc.grm(X, a), K)

    def _llik(self, y, eta):
        """Return E_q[ln p(y | eta)] assuming a logit link."""
        F = y * eta - T.nnet.softplus(eta)
        return T.mean(T.sum(F, axis=1)) - T.mean(F)

class GaussianModel(Model):
    def __init__(self, X, y, a, K=None):
        self.X = X
        self.y = y
        self.a = a
        self.var_x = X.var(axis=0).sum()
        self.pve = ctra.pcgc.estimate(y, ctra.pcgc.grm(X, a), K)

    def _log_weight(self, pi, tau, alpha=None, beta=None, atol=1e-4, **hyperparams):
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
        if alpha is None:
            alpha = 0.5 * numpy.ones(p)
            alpha /= alpha.sum()
        if beta is None:
            beta = R.normal(size=p)
        sigma2 = y.var()
        print({'pve': self.pve, 'pi': pi, 'tau': tau, 'sigma2': sigma2, 'var_x': self.var_x})
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
        print('{:>8s} {:>8s} {:>8s}'.format('ELBO', 'sigma2', 'Vars'), file=sys.stderr)
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
            print('{:>8.6g} {:>8.3g} {:>8.0f}'.format(elbo, sigma2, alpha.sum()), file=sys.stderr)
            if elbo > 0:
                raise ValueError('ELBO must be non-positive')
            if elbo < elbo_:
                converged = True
            elif numpy.isclose(alpha_, alpha, atol=atol).all() and numpy.isclose(alpha_ * beta_, alpha * beta, atol=atol).all():
                converged = True
        return elbo, alpha, beta
