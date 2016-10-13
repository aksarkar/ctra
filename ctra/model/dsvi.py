"""Doubly stochastic variational inference

We cannot write an analytical solution for the variational approximation, so we
take a doubly stochastic approach, using Monte Carlo integration to estimate
intractable expectations (re-parameterizing integrals as sums) and drawing
samples (due to the non-conjugate prior) to estimate the gradient. We
additionally use a control variate to reduce the variance of the gradient
estimator.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import logging

import numpy
import scipy.misc
import scipy.special
import theano
import theano.tensor as T

from .base import ImportanceSampler

logger = logging.getLogger(__name__)

_real = theano.config.floatX
_F = theano.function
_S = lambda x: theano.shared(x, borrow=True)
_Z = lambda n: numpy.zeros(n).astype(_real)
_R = numpy.random
_N = lambda n: _R.normal(size=n).astype(_real)

class DSVI(ImportanceSampler):
    """Class providing the implementation of the optimizer

    This is intended to provide a pickle-able object to re-use the Theano
    compiled function across hyperparameter samples.

    """
    def __init__(self, X_, y_, a_, pve, minibatch_n=None, stoch_samples=None,
                 learning_rate=None, *params, **kwargs):
        """Compile the Theano function which takes a gradient step"""
        super().__init__(X_, y_, a_, pve)

        logger.debug('Building the Theano graph')
        # Observed data
        n, p = X_.shape
        X = _S(X_.astype(_real))
        y = _S(y_.astype(_real))
        a = _S(a_.astype('int8'))

        # Hyperparameters
        pi = T.vector(name='pi')
        pi_deref = _S(_Z(p))
        tau = T.vector(name='tau')
        tau_deref = _S(_Z(p))

        # Variational parameters
        alpha_raw = _S(_Z(p))
        alpha = T.cast(T.clip(T.nnet.sigmoid(alpha_raw), self.eps, 1 - self.eps), _real)
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
        if stoch_samples is None and minibatch_n > 10:
            stoch_samples = 1
        else:
            stoch_samples = 10
        eta_raw = _S(numpy.random.normal(size=(stoch_samples, minibatch_n)).astype(_real))
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
        self._randomize = _F(inputs=[pi, tau], outputs=[],
                             updates=[(alpha_raw, _Z(p)), (beta, _N(p)),
                                      (pi_deref, T.basic.choose(a, pi)),
                                      (tau_deref, T.basic.choose(a, tau))])
        alpha_ = T.vector()
        beta_ = T.vector()
        self._initialize = _F(inputs=[alpha_, beta_, pi, tau], outputs=[],
                              updates=[(alpha_raw, alpha_), (beta, beta_),
                                       (pi_deref, T.basic.choose(a, pi)),
                                       (tau_deref, T.basic.choose(a, tau))],
                              allow_input_downcast=True)

        grad = T.grad(elbo, self.params)
        if learning_rate is None:
            learning_rate = numpy.array(5e-2 / minibatch_n, dtype=_real)
        logger.debug('Minibatch size = {}'.format(minibatch_n))
        logger.debug('Initial learning rate = {}'.format(learning_rate))
        self.vb_step = _F(inputs=[epoch],
                          outputs=elbo,
                          updates=[(p_, T.cast(p_ + 10 ** -(epoch // 1e5) * learning_rate * (g - cv), _real))
                                   for p_, g, cv in zip(self.params, grad, control)])

        self._opt = _F(inputs=[], outputs=[alpha, beta])
        logger.debug('Finished initializing')

    def _llik(self, *args):
        raise NotImplementedError

    def _log_weight(self, params=None, weight=0.5, poll_iters=1000,
                    min_iters=100000, atol=1, true_causal=None, **hyperparams):
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
            self._randomize(**hyperparams)
        else:
            alpha, beta = params
            self._initialize(scipy.special.expit(alpha), beta, **hyperparams)
        converged = False
        t = 0
        ewma = 0
        ewma_ = ewma
        while not converged:
            t += 1
            elbo = self.vb_step(epoch=t)
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
                    alpha, beta = self._opt()
        return ewma, (alpha, beta)

class GaussianDSVI(DSVI):
    def __init__(self, X, y, a, pve, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        _sigma2 = theano.shared(numpy.array([pve.sum() * y.var()],
                                            dtype=_real))
        self.sigma2 = T.addbroadcast(_sigma2, 0)
        logger.debug('Fixing sigma2 to {}'.format(_sigma2.get_value()))
        super().__init__(X, y, a, pve, params=[self.sigma2], **kwargs)

    def _llik(self, y, eta):
        """Return E_q[ln p(y | eta, theta_0)] assuming a linear link."""
        F = -.5 * (T.log(self.sigma2) + T.sqr(y - eta) / self.sigma2)
        return T.mean(T.sum(F, axis=1))

class LogisticDSVI(DSVI):
    def __init__(self, X, y, a, pve, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        _bias = theano.shared(numpy.array([0], dtype=_real))
        self.bias = T.addbroadcast(_bias, 0)
        super().__init__(X, y, a, pve, params=[self.bias], **kwargs)
        # Now add terms to ELBO for the bias: q(bias) ~ N(bias; theta_0,
        # 2.5^2), q(theta_0) ~ N(0, 2.5^2)
        self._elbo += self.bias / 2.5

    def _llik(self, y, eta):
        """Return E_q[ln p(y | eta, theta_0)] assuming a logit link."""
        F = y * (eta + self.bias) - T.nnet.softplus(eta + self.bias)
        return T.mean(T.sum(F, axis=1))
