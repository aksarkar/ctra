"""Variational auto-encoder

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import logging

import numpy
import scipy.misc
import scipy.special
import theano
import theano.tensor as T

from matplotlib.pyplot import *
from .base import Algorithm

logger = logging.getLogger(__name__)

_real = theano.config.floatX
_F = theano.function
_Z = lambda n: numpy.zeros(n).astype(_real)

def _S(x, **kwargs):
    return theano.shared(x, borrow=True, **kwargs)

class VAE(Algorithm):
    """Base class providing the generic implementation. Specialized sub-classes are
needed for specific likelihoods.

    """
    def __init__(self, X_, y_, a_, stoch_samples=50, learning_rate=1e-4,
                 hyperparam_means=None, hyperparam_log_precs=None,
                 random_state=None, minibatch_n=None, **kwargs):
        """Compile the Theano function which takes a gradient step"""
        super().__init__(X_, y_, a_, None)
        self.max_prec = 1e3

        logger.debug('Building the Theano graph')

        # Observed data. This needs to be symbolic for minibatches
        # TODO: borrow, GPU transfer, HDF5 transfer
        X = T.fmatrix(name='X')
        y = T.fvector(name='y')
        a = T.ivector(name='a')

        # Variational surrogate for target hyperposterior
        m = self.p.shape[0]
        q_logit_pi_mean = _S(_Z(m), name='q_logit_pi_mean')
        q_logit_pi_log_prec = _S(_Z(m), name='q_logit_pi_log_prec')
        q_log_tau_mean = _S(_Z(m), name='q_log_tau_mean')
        q_log_tau_log_prec = _S(_Z(m), name='q_log_tau_log_prec')

        # Variational parameters
        n, p = X_.shape
        q_logit_z = _S(_Z(p), name='q_logit_z')
        q_z = T.nnet.sigmoid(q_logit_z)
        q_theta_mean = _S(_Z(p), name='q_theta_mean')
        q_theta_log_prec = _S(_Z(p), name='q_theta_log_prec')
        self.min_prec = 1e-3
        q_theta_prec = self.min_prec + T.nnet.softplus(q_theta_log_prec)
        self.params = [q_logit_z, q_theta_mean, q_theta_log_prec]

        # These will include model-specific terms. Assume everything is
        # Gaussian on the variational side to simplify
        self.hyperparam_means = [q_logit_pi_mean, q_log_tau_mean]
        self.hyperparam_log_precs = [q_logit_pi_log_prec, q_log_tau_log_prec]
        if hyperparam_means is not None:
            self.hyperparam_means.extend(hyperparam_means)
        if hyperparam_log_precs is not None:
            self.hyperparam_log_precs.extend(hyperparam_log_precs)

        self.hyperprior_means = [numpy.repeat(-2, m).astype(_real), _Z(m)]
        self.hyperprior_log_precs = [_Z(m), _Z(m)]
        for _ in hyperparam_means:
            self.hyperprior_means.append(_Z(m))
            self.hyperprior_log_precs.append(_Z(m))

        # We need to perform inference on minibatches of samples for speed. Rather
        # than taking balanced subsamples, we take a sliding window over a
        # permutation which is balanced in expectation.
        epoch = T.iscalar(name='epoch')

        # Pre-generate stochastic samples
        if random_state is None:
            _R = numpy.random
        else:
            _R = random_state
        noise = _S(_R.normal(size=(5 * stoch_samples, n)).astype(_real), name='noise')

        # Re-parameterize eta = X theta (Kingma, Salimans, & Welling NIPS
        # 2015), and backpropagate through the RNG (Kingma & Welling, ICLR
        # 2014).
        eta_mean = T.dot(X, q_z * q_theta_mean)
        eta_var = T.dot(T.sqr(X), q_z / q_theta_prec + q_z * (1 - q_z) * T.sqr(q_theta_mean))
        eta_minibatch = epoch % 5
        eta_raw = noise[eta_minibatch * stoch_samples:(eta_minibatch + 1) * stoch_samples]
        eta = eta_mean + T.sqrt(eta_var) * eta_raw

        # We need to generate independent noise samples for the hyperparameters
        phi_minibatch = (epoch + 1) % 5
        phi_raw = noise[phi_minibatch * stoch_samples:(phi_minibatch + 1) * stoch_samples,:m]

        # We need to warm up the objective function in order to avoid
        # degenerate solutions where all variational free parameters go to
        # zero. The idea is given in Sønderby et al., NIPS 2016
        error = self._llik(self.y, eta, phi_raw) * self.scale_n
        # We don't need to use the hyperparameter noise samples for these
        # parameters because we can deal with them analytically
        pi = T.addbroadcast(T.nnet.sigmoid(q_logit_pi_mean), 0)
        tau = T.addbroadcast(T.exp(q_log_tau_mean), 0)
        # Rasmussen and Williams, Eq. A.23, conditioning on q_z (alpha in our
        # notation)
        kl_qtheta_ptheta = .5 * T.sum(q_z * (1 + T.log(tau) - T.log(q_theta_prec) + tau * (T.sqr(q_theta_mean) + 1 / q_theta_prec)))
        # Rasmussen and Williams, Eq. A.22
        kl_qz_pz = T.sum(q_z * T.log(q_z / pi) + (1 - q_z) * T.log((1 - q_z) / (1 - pi)))
        kl_hyper = 0
        for mean, log_prec, prior_mean, prior_log_prec in zip(self.hyperparam_means, self.hyperparam_log_precs, self.hyperprior_means, self.hyperprior_log_precs):
            prec = self.min_prec + T.nnet.softplus(log_prec)
            prior_prec = self.min_prec + T.nnet.softplus(prior_log_prec)
            kl_hyper += .5 * T.sum(1 + T.log(prior_prec) - T.log(prec) + prior_prec * (T.sqr(mean - prior_mean) + 1 / prec))
        kl = kl_qtheta_ptheta + kl_qz_pz + kl_hyper
        elbo = error - temperature * kl

        logger.debug('Compiling the Theano functions')
        init_updates = [(param, _Z(p)) for param in self.params]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_means, self.hyperprior_means)]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_log_precs, self.hyperprior_log_precs)]
        self.initialize = _F(inputs=[], outputs=[], updates=init_updates)

        self.variational_params = self.params + self.hyperparam_means + self.hyperparam_log_precs
        all_params = self.params + self.hyperparam_means + self.hyperparam_log_precs
        grad = T.grad(elbo, self.variational_params)
        sgd_updates = [(param, param + T.cast(learning_rate, dtype=_real) * g)
                       for param, g in zip(self.variational_params, grad)]
        sgd_givens = {X: X_.astype(_real), y: y_.astype(_real), a: a_}
        self.sgd_step = _F(inputs=[epoch], outputs=[elbo], updates=sgd_updates, givens=sgd_givens)
        self._trace = _F(inputs=[epoch], outputs=[epoch, elbo, error, kl_qz_pz, kl_qtheta_ptheta, kl_hyper] + all_params, givens=sgd_givens)
        self._trace_grad = _F(inputs=[epoch], outputs=T.grad(elbo, self.variational_params), givens=sgd_givens)
        opt_outputs = [elbo, q_z, q_theta_mean, T.nnet.sigmoid(q_logit_pi_mean), T.exp(q_log_tau_mean)]
        self.opt = _F(inputs=[epoch], outputs=opt_outputs, givens=sgd_givens)
        logger.debug('Finished initializing')

    def _llik(self, *args):
        raise NotImplementedError

    def log_weight(self, *args):
        raise NotImplementedError
        
    def fit(self, atol=1e-3, **kwargs):
        self.initialize()
        t = 0
        elbo_ = float('-inf')
        self.trace = []
        self.trace_grad = []
        while t < 1000:
            t += 1
            self.trace.append(self._trace(t))
            self.trace_grad.append(self._trace_grad(t))
            elbo = self.sgd_step(epoch=t)
            logger.debug('Epoch {}: {}'.format(t, elbo))
            if not numpy.isfinite(elbo):
                return self
        logger.info('Converged at epoch {}'.format(t))
        self._evidence, self.pip, self.q_theta_mean, self.pi, self.tau = self.opt(epoch=t)
        self.theta = self.pip * self.q_theta_mean
        return self

    def predict(self, x):
        """Return the posterior mean prediction"""
        return x.dot(self.theta)

    def score(self, x, y):
        """Return the coefficient of determination of the model fit"""
        return 1 - (numpy.square(self.predict(x) - y).sum() /
                    numpy.square(y - y.mean()).sum())

class GaussianVAE(VAE):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.log_sigma2_mean = _S(_Z(1))
        log_sigma2_log_prec = _S(_Z(1))
        self.log_sigma2_prec = 1e-3 + T.nnet.softplus(log_sigma2_log_prec)
        super().__init__(X, y, a,
                         hyperparam_means=[self.log_sigma2_mean],
                         hyperparam_log_precs=[log_sigma2_log_prec],
                         **kwargs)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, theta_0)] assuming a linear link."""
        phi = T.exp(T.addbroadcast(self.min_prec + T.nnet.softplus(self.log_sigma2_mean + T.sqrt(1 / self.log_sigma2_prec) * phi_raw), 1))
        F = -.5 * (T.log(phi) + T.sqr(y - eta) / phi)
        return T.mean(T.sum(F, axis=1))

class LogisticVAE(VAE):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.bias_mean = _S(_Z(1))
        bias_log_prec = _S(_Z(1))
        self.bias_prec = T.nnet.softplus(bias_log_prec)
        super().__init__(X, y, a, hyperparam_means=[self.bias_mean],
                         hyperparam_log_precs=[bias_log_prec],
                         **kwargs)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, theta_0)] assuming a logit link."""
        phi = T.addbroadcast(T.nnet.softplus(self.bias_mean + T.sqrt(1 / self.bias_prec) * phi_raw), 1)
        F = y * (eta + phi) - T.nnet.softplus(eta + phi)
        return T.mean(T.sum(F, axis=1))

    def predict(self, x):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        # This depends on local Theano tensors, so compile it here
        self.predict = _F(inputs=[self.X], outputs=[T.nnet.sigmoid(T.dot(self.X, self.theta) + T.addbroadcast(self.bias_mean, 0))])
        return self

    def score(self, x, y):
        yhat = (numpy.array(self.predict(x)) > 0.5)
        return numpy.asscalar((y == yhat).sum() / y.shape[0])
