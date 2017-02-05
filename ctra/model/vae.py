"""Variational auto-encoder

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

class VAE(Algorithm):
    """Base class providing the generic implementation. Specialized sub-classes are
needed for specific likelihoods.

    """
    def __init__(self, X_, y_, a_, stoch_samples=10, learning_rate=1e-4,
                 warmup_rate=1e-3, hyperparam_means=None,
                 hyperparam_logit_precs=None, **kwargs):
        """Compile the Theano function which takes a gradient step"""
        super().__init__(X_, y_, a_, None)

        logger.debug('Building the Theano graph')

        # Observed data. This needs to be symbolic for minibatches
        # TODO: borrow, GPU transfer, HDF5 transfer
        X = T.matrix(name='X')
        y = T.vector(name='y')
        a = T.ivector(name='a')

        # Variational parameters
        n, p = X_.shape
        q_logit_z = _S(_Z(p))
        q_z = T.cast(T.clip(T.nnet.sigmoid(q_logit_z), self.eps, 1 - self.eps), _real)
        q_theta_mean = _S(_N(p))
        q_theta_logit_prec = _S(_Z(p))
        q_theta_prec = 1e5 * T.nnet.sigmoid(q_theta_logit_prec)
        self.params = [q_logit_z, q_theta_mean, q_theta_logit_prec]

        # Variational surrogate for target hyperposterior
        m = self.p.shape[0]
        q_logit_pi_mean = _S(_Z(m))
        q_logit_pi_logit_prec = _S(_Z(m))
        q_log_tau_mean = _S(_Z(m))
        q_log_tau_logit_prec = _S(_Z(m))

        # These will include model-specific terms. Assume everything is
        # Gaussian on the variational side to simplify
        self.hyperparam_means = [q_logit_pi_mean, q_log_tau_mean]
        if hyperparam_means is not None:
            self.hyperparam_means.extend(hyperparam_means)
        self.hyperparam_logit_precs = [q_logit_pi_logit_prec, q_log_tau_logit_prec]
        if hyperparam_logit_precs is not None:
            self.hyperparam_logit_precs.extend(hyperparam_logit_precs)

        # We need to perform inference on minibatches of samples for speed. Rather
        # than taking balanced subsamples, we take a sliding window over a
        # permutation which is balanced in expectation.
        epoch = T.iscalar(name='epoch')

        # Pre-generate stochastic samples
        noise = _S(numpy.random.normal(size=(5 * stoch_samples, n)).astype(_real))

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
        phi = [mean + T.sqrt(1e-4 / T.nnet.sigmoid(logit_prec)) * phi_raw
               for mean, logit_prec in zip(self.hyperparam_means, self.hyperparam_logit_precs)]

        # We need to warm up the objective function in order to avoid
        # degenerate solutions where all variational free parameters go to
        # zero. The idea is given in Sønderby et al., NIPS 2016
        temperature = T.clip(T.cast(warmup_rate * epoch, _real), 0, 1)
        error = self._llik(y, eta, phi_raw)
        kl = T.mean(.5 * T.sum(q_z * (1 + T.log(phi[1]) - T.log(q_theta_prec) -
                                      phi[1] * (T.sqr(q_theta_mean) + 1 / q_theta_prec)))
                    - T.sum(q_z * T.log(q_z / phi[0]) + (1 - q_z) * T.log((1 - q_z) / (1 - phi[0]))))
        for mean, prec in zip(self.hyperparam_means, self.hyperparam_logit_precs):
            kl += -.5 * T.sum(prec * (T.sqr(mean) + 1 / prec))
        elbo = error + temperature * kl

        logger.debug('Compiling the Theano functions')
        init_updates = [(param, _Z(p)) for param in self.params]
        init_updates += [(param, _Z(m)) for param in self.hyperparam_means]
        init_updates += [(param, _Z(m)) for param in self.hyperparam_logit_precs]
        self.initialize = _F(inputs=[], outputs=[], updates=init_updates)

        variational_params = self.params + self.hyperparam_means + self.hyperparam_logit_precs 
        grad = T.grad(elbo, variational_params)
        sgd_updates = [(param, param + numpy.array(learning_rate, dtype=_real) * g)
                       for param, g in zip(variational_params, grad)]
        sgd_givens = {X: X_.astype(_real), y: y_.astype(_real), a: a_}
        self.sgd_step = _F(inputs=[epoch], outputs=elbo, updates=sgd_updates, givens=sgd_givens)
        self.trace = _F(inputs=[epoch], outputs=[error, kl, q_logit_pi_mean, q_z.max()], givens=sgd_givens)
        self.opt = _F(inputs=[epoch], outputs=[elbo, q_logit_pi_mean, q_log_tau_mean, q_z, q_theta_mean], givens=sgd_givens)
        logger.debug('Finished initializing')

    def _llik(self, *args):
        raise NotImplementedError

    def log_weight(self, *args):
        raise NotImplementedError
        
    def fit(self, **kwargs):
        self.initialize()
        t = 0
        elbo_ = float('-inf')
        while t < 10000:
            t += 1
            elbo = self.sgd_step(epoch=t)
            assert numpy.isfinite(elbo)
            if ((self.warmup_rate == 0 or t > 1 / self.warmup_rate) and
                (elbo < elbo_ or numpy.isclose(elbo, elbo_, atol=1e-2))):
                break
            else:
                logger.debug(self.trace(t))
                elbo_ = elbo
        logger.debug('Converged at epoch {}'.format(t))
        self._evidence, self.pi, self.tau, self.pip, self.q_theta_mean = self.opt(epoch=t)
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
    def __init__(self, X, y, a, pve, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.log_sigma2_mean = _S(numpy.atleast_1d(numpy.log(y.var())).astype(_real))
        log_sigma2_logit_prec = _S(_Z(1))
        self.log_sigma2_prec = 1e4 * T.nnet.sigmoid(log_sigma2_logit_prec)
        super().__init__(X, y, a, hyperparam_means=[self.log_sigma2_mean],
                         hyperparam_logit_precs=[log_sigma2_logit_prec],
                         **kwargs)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, theta_0)] assuming a linear link."""
        phi = T.addbroadcast(self.log_sigma2_mean + T.sqrt(1 / self.log_sigma2_prec) * phi_raw.T, 1)
        F = -.5 * (T.log(phi) + T.sqr(y - eta) / phi)
        return T.mean(T.sum(F, axis=1))

class LogisticVAE(VAE):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        bias_mean = _S(_Z(1))
        bias_logit_prec = _S(_Z(1))
        bias_prec = 1e4 * T.nnet.sigmoid(bias_logit_prec)
        super().__init__(X, y, a, hyperparam_means=[bias_mean],
                         hyperparam_logit_precs=[bias_logit_prec],
                         **kwargs)

    def _llik(self, y, eta, phi):
        """Return E_q[ln p(y | eta, theta_0)] assuming a logit link."""
        F = y * (eta + phi) - T.nnet.softplus(eta + phi)
        return T.mean(T.sum(F, axis=1))
