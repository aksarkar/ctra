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

from .base import Algorithm

logger = logging.getLogger(__name__)

_real = theano.config.floatX
_F = theano.function
_S = lambda x: theano.shared(x, borrow=True)
_Z = lambda n: numpy.zeros(n).astype(_real)
_R = numpy.random
_N = lambda n: _R.normal(size=n).astype(_real)

class DSVI(Algorithm):
    """Class providing the implementation of the optimizer

    This is intended to provide a pickle-able object to re-use the Theano
    compiled function across hyperparameter samples.

    """
    def __init__(self, X_, y_, a_, pve, minibatch_n=None, stoch_samples=10,
                 learning_rate=1e-3, params=None, **kwargs):
        """Compile the Theano function which takes a gradient step"""
        super().__init__(X_, y_, a_, pve)

        logger.debug('Building the Theano graph')
        # Observed data
        n, p = X_.shape
        self.X = _S(X_)
        self.y = _S(y_)
        self.a = _S(a_)

        # Minibatch
        if minibatch_n is None:
            minibatch_n = n
        self.minibatch_n = minibatch_n
        self.scale_n = n // minibatch_n
        X = T.matrix()
        y = T.vector()
        a = T.bvector()

        # Hyperparameters
        pi = T.vector(name='pi')
        pi_deref = _S(_Z(p))
        tau = T.vector(name='tau')
        tau_deref = _S(_Z(p))

        # Variational parameters
        alpha_raw = _S(_Z(p))
        alpha = T.nnet.sigmoid(alpha_raw)
        beta = _S(_N(p))
        gamma_raw = _S(_Z(p))
        gamma = 1e-3 + T.nnet.softplus(gamma_raw)

        self.params = [alpha_raw, beta, gamma_raw]
        if params is None:
            params = []
        self.params.extend(params)

        # We need to perform inference on minibatches of samples for speed. Rather
        # than taking balanced subsamples, we take a sliding window over a
        # permutation which is balanced in expectation.
        epoch = T.iscalar(name='epoch')

        # Variational approximation (re-parameterize eta = X theta). This is a
        # "Gaussian reconstruction" in that we characterize its expectation and
        # variance, then approximate its distribution with a Gaussian.
        #
        # We need to take the gradient of an intractable integral, so we re-write
        # it as a Monte Carlo integral which is differentiable, following Kingma &
        # Welling, ICLR 2014 (http://arxiv.org/abs/1312.6114).
        mu = T.dot(X, alpha * beta)
        nu = T.dot(T.sqr(X), alpha / gamma + alpha * (1 - alpha) * T.sqr(beta))

        # Pre-compute the SGVB samples
        noise = _S(_R.normal(size=(5 * stoch_samples, minibatch_n)).astype(_real))
        noise_minibatch = epoch % 5
        eta_raw = noise[noise_minibatch * stoch_samples:(noise_minibatch + 1) * stoch_samples]
        eta = mu + T.sqrt(nu) * eta_raw

        # Objective function
        if 'elbo' not in self.__dict__:
            # This is a hack to allow subclasses to modify the graph before we
            # compile the Theano functions
            self.elbo = T.as_tensor_variable(0)
        kl_hyper = -self.elbo
        # The log likelihood is for the minibatch, but we need to scale up
        # to the full dataset size
        error = self._llik(y, eta) * self.scale_n
        # KL(q(theta) || p(theta))
        kl_theta = .5 * T.sum(alpha * (1 - T.log(tau_deref) + T.log(gamma) + tau_deref * (T.sqr(beta) + 1 / gamma)))
        # KL(q(z) || p(z))
        kl_z = T.sum(alpha * T.log(alpha / pi_deref) + (1 - alpha) * T.log((1 - alpha) / (1 - pi_deref)))
        self.elbo += error - kl_theta - kl_z

        logger.debug('Compiling the Theano functions')
        init_updates = [(alpha_raw, _Z(p)), (beta, _Z(p)), (gamma_raw, _Z(p)),
                        (pi_deref, T.basic.choose(a, pi)),
                        (tau_deref, T.basic.choose(a, tau))]
        # This is needed to re-initialize logistic regression bias
        init_updates += [(param, 0) for param in params]
        self._initialize = _F(inputs=[pi, tau], outputs=[], updates=init_updates, givens=[(a, self.a)])
        grad = T.grad(self.elbo, self.params)
        sgd_updates = [(param, param + learning_rate * g)
                       for param, g in zip(self.params, grad)]
        sample_minibatch = epoch % (n // minibatch_n)
        sgd_givens = [(X, self.X[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]),
                      (y, self.y[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]),
                      (a, self.a)]
        self.vb_step = _F(inputs=[epoch], outputs=self.elbo, updates=sgd_updates, givens=sgd_givens)

        self._trace = _F(inputs=[epoch], outputs=[self.elbo, error, kl_theta, kl_z, kl_hyper, alpha, beta, gamma] + params, givens=sgd_givens)

        self._opt = _F(inputs=[], outputs=[alpha, beta])
        logger.debug('Finished initializing')

    def _llik(self, *args):
        raise NotImplementedError

    def log_weight(self, pi, tau, max_epochs=4000, true_causal=None, **kwargs):
        """Return optimum ELBO and variational parameters which achieve it.

        weight - weight for exponential moving average of ELBO
        poll_iters - number of iterations before polling objective function
        min_iters - minimum number of iterations
        atol - maximum change in objective for convergence
        hyperparams - pi, tau, etc.

        """
        hyperparams = {'pi': pi, 'tau': tau}
        logger.debug('Starting SGD given {}'.format(hyperparams))
        # Re-initialize, otherwise everything breaks
        self._initialize(**hyperparams)
        self.trace = []
        t = 0
        while t < max_epochs * self.scale_n:
            self.trace.append(self._trace(t))
            if not t % 1000:
                logger.debug('\t'.join('{:.3g}'.format(numpy.asscalar(x)) for x in self.trace[-1][:5]))
            t += 1
            elbo = self.vb_step(epoch=t)
            if not numpy.isfinite(elbo):
                import pdb; pdb.set_trace()
            elif elbo >= 0:
                import pdb; pdb.set_trace()
        return elbo, self._opt()

class GaussianDSVI(DSVI):
    def __init__(self, X, y, a, pve, **kwargs):
        super().__init__(X, y, a, pve, **kwargs)

    def _llik(self, y, eta):
        """Return E_q[ln p(y | eta, theta_0)] assuming a linear link.

        Fix sigma^2 to 1. Empirically, this works although we have no proof.

        """
        F = -.5 * T.sqr(y - eta)
        return T.mean(T.sum(F, axis=1))

class LogisticDSVI(DSVI):
    def __init__(self, X, y, a, pve, K, P, **kwargs):
        # Neuhaus, 2002 (eq. 3)
        self.rate_ratio = P * (1 - K) / (K * (1 - P))
        logger.debug('Rate ratio = {:.3g}'.format(self.rate_ratio))
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.bias_mean = theano.shared(numpy.array(0, dtype=_real))
        self.bias_log_prec = theano.shared(numpy.array(0, dtype=_real))
        self.bias_prec = 1e-3 + T.nnet.softplus(self.bias_log_prec)
        # Variational surrogate for the bias term. p(bias) = N(0, 1); q(bias) = N(m, v^-1)
        self.elbo = -.5 * (1 - 1 + T.log(self.bias_prec) + T.sqr(self.bias_mean) + 1 / self.bias_prec)
        super().__init__(X, y, a, pve, params=[self.bias_mean, self.bias_log_prec], **kwargs)

    def _link(self, eta):
        # Neuhaus, 2002 (eq. 3-5)
        mu = T.nnet.sigmoid(eta + self.bias_mean)
        return self.rate_ratio * mu / (1 + mu * (self.rate_ratio - 1))

    def _llik(self, y, eta):
        """Return E_q[ln p(y | eta, theta_0)] assuming a logit link."""
        F = y * T.log(self._link(eta)) + (1 - y) * T.log(1 - self._link(eta))
        return T.mean(T.sum(F, axis=1))
