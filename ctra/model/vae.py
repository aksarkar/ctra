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

        logger.debug('Building the Theano graph')

        # Observed data. This needs to be symbolic for minibatches
        # TODO: borrow, GPU transfer, HDF5 transfer
        self.X_ = _S(X_.astype(_real))
        self.y_ = _S(y_.astype(_real))

        # One-hot encode the annotations
        n, p = X_.shape
        m = self.p.shape[0]
        A = numpy.zeros((p, m)).astype('i1')
        A[range(p), a_] = 1
        self.A = _S(A)

        self.X = T.fmatrix(name='X')
        self.y = T.fvector(name='y')

        if minibatch_n is None:
            minibatch_n = n
        self.scale_n = n / minibatch_n

        # Variational surrogate for target hyperposterior
        self.q_probit_pi_mean = _S(_Z(m), name='q_probit_pi_mean')
        self.q_probit_pi_log_prec = _S(_Z(m), name='q_probit_pi_log_prec')
        self.q_pi_mean = .5 * (1 + T.erf(self.q_probit_pi_mean / (2 * T.sqrt(2))))
        self.q_log_tau_mean = _S(_Z(m), name='q_log_tau_mean')
        self.q_log_tau_log_prec = _S(_Z(m), name='q_log_tau_log_prec')
        self.q_tau_mean = T.exp(self.q_log_tau_mean)

        # We don't need to use the hyperparameter noise samples for these
        # parameters because we can deal with them analytically
        pi = T.dot(self.A, self.q_pi_mean)
        tau = T.dot(self.A, self.q_tau_mean)

        # Variational parameters
        self.q_logit_z = _S(_Z(p), name='q_logit_z')
        self.q_z = T.nnet.sigmoid(self.q_logit_z)
        self.q_theta_mean = _S(_Z(p), name='q_theta_mean')
        self.q_theta_log_prec = _S(_Z(p), name='q_theta_log_prec')
        self.min_prec = 1e-3
        self.q_theta_prec = self.min_prec + T.nnet.softplus(self.q_theta_log_prec)
        self.params = [self.q_logit_z, self.q_theta_mean, self.q_theta_log_prec]

        # These will include model-specific terms. Assume everything is
        # Gaussian on the variational side to simplify
        self.hyperparam_means = [self.q_probit_pi_mean, self.q_log_tau_mean]
        self.hyperparam_log_precs = [self.q_probit_pi_log_prec, self.q_log_tau_log_prec]
        if hyperparam_means is not None:
            self.hyperparam_means.extend(hyperparam_means)
        if hyperparam_log_precs is not None:
            self.hyperparam_log_precs.extend(hyperparam_log_precs)

        self.hyperprior_means = [_Z(m), _Z(m)]
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
        noise = _S(_R.normal(size=(5 * stoch_samples, minibatch_n)).astype(_real), name='noise')

        # Re-parameterize eta = X theta (Kingma, Salimans, & Welling NIPS
        # 2015), and backpropagate through the RNG (Kingma & Welling, ICLR
        # 2014).
        self.eta_mean = T.dot(self.X, self.q_z * self.q_theta_mean)
        eta_var = T.dot(T.sqr(self.X), self.q_z / self.q_theta_prec + self.q_z * (1 - self.q_z) * T.sqr(self.q_theta_mean))
        eta_minibatch = epoch % 5
        eta_raw = noise[eta_minibatch * stoch_samples:(eta_minibatch + 1) * stoch_samples]
        eta = self.eta_mean + T.sqrt(eta_var) * eta_raw

        # We need to generate independent noise samples for the hyperparameters
        phi_minibatch = (epoch + 1) % 5
        phi_raw = noise[phi_minibatch * stoch_samples:(phi_minibatch + 1) * stoch_samples,:m]

        # We need to warm up the objective function in order to avoid
        # degenerate solutions where all variational free parameters go to
        # zero. The idea is given in Sønderby et al., NIPS 2016
        error = self._llik(self.y, eta, phi_raw)

        # Rasmussen and Williams, Eq. A.23, conditioning on q_z (alpha in our
        # notation)
        kl_qtheta_ptheta = .5 * T.sum(self.q_z * (1 - T.log(tau) + T.log(self.q_theta_prec) + tau * (T.sqr(self.q_theta_mean) + 1 / self.q_theta_prec)))
        # Rasmussen and Williams, Eq. A.22
        kl_qz_pz = T.sum(self.q_z * T.log(self.q_z / pi) + (1 - self.q_z) * T.log((1 - self.q_z) / (1 - pi)))
        kl_hyper = 0
        for mean, log_prec, prior_mean, prior_log_prec in zip(self.hyperparam_means, self.hyperparam_log_precs, self.hyperprior_means, self.hyperprior_log_precs):
            prec = self.min_prec + T.nnet.softplus(log_prec)
            prior_prec = self.min_prec + T.nnet.softplus(prior_log_prec)
            kl_hyper += .5 * T.sum(1 - T.log(prior_prec) + T.log(prec) + prior_prec * (T.sqr(mean - prior_mean) + 1 / prec))
        kl = kl_qtheta_ptheta + kl_qz_pz + kl_hyper
        # Kingma & Welling 2013 (eq. 8)
        elbo = (error - kl) * self.scale_n

        logger.debug('Compiling the Theano functions')
        init_updates = [(self.q_logit_z, _R.normal(size=p).astype(_real))]
        init_updates += [(param, _Z(p)) for param in self.params[1:]]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_means, self.hyperprior_means)]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_log_precs, self.hyperprior_log_precs)]
        self.initialize = _F(inputs=[], outputs=[], updates=init_updates)

        self.variational_params = self.params + self.hyperparam_means + self.hyperparam_log_precs
        all_params = self.params + self.hyperparam_means + self.hyperparam_log_precs
        grad = T.grad(elbo, self.variational_params)
        sgd_updates = [(param, param + T.cast(learning_rate, dtype=_real) * g)
                       for param, g in zip(self.variational_params, grad)]
        sample_minibatch = epoch % (n // minibatch_n)
        sgd_givens = {self.X: self.X_[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n],
                      self.y: self.y_[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]}
        self.sgd_step = _F(inputs=[epoch], outputs=elbo, updates=sgd_updates, givens=sgd_givens)
        self._trace = _F(inputs=[epoch], outputs=[epoch, elbo, error, kl_qz_pz, kl_qtheta_ptheta, kl_hyper] + all_params, givens=sgd_givens)
        self._trace_grad = _F(inputs=[epoch], outputs=T.grad(elbo, self.variational_params), givens=sgd_givens)
        opt_outputs = [elbo, self.q_z, self.q_theta_mean, self.q_pi_mean, self.q_tau_mean]
        self.opt = _F(inputs=[epoch], outputs=opt_outputs, givens=sgd_givens)
        logger.debug('Finished initializing')

    def _llik(self, *args):
        raise NotImplementedError

    def log_weight(self, *args):
        raise NotImplementedError
        
    def fit(self, max_iters=1000, xv=None, yv=None, trace=False, **kwargs):
        logger.debug('Starting SGD')
        self.initialize()
        t = 0
        elbo_ = float('-inf')
        loss = float('inf')
        self.trace = []
        while t < max_iters * self.scale_n:
            t += 1
            if trace:
                self.trace.append(self._trace(t))
            else:
                self.trace = [self._trace(t)]
            elbo = self.sgd_step(epoch=t)
            if not t % (100 * self.scale_n):
                validation_loss = self.loss(xv, yv)
                if elbo < elbo_:
                    logger.warn('ELBO increased, stopping early')
                    break
                elbo_ = elbo
                loss = validation_loss
                outputs = self.trace[-1][:6]
                outputs.append(self.score(self.X_.get_value(), self.y_.get_value()))
                outputs.append(self.score(xv, yv))
                logger.debug('\t'.join('{:.3g}'.format(numpy.asscalar(x)) for x in outputs))
            if not numpy.isfinite(elbo):
                logger.warn('ELBO infinite. Stopping early')
                break
        self._evidence, self.pip, self.q_theta_mean, self.pi, self.tau = self.opt(epoch=t)
        logger.info('Converged at epoch {}'.format(t))
        self.theta = self.pip * self.q_theta_mean
        return self

    def predict(self, x):
        """Return the posterior mean prediction"""
        raise NotImplementedError

    def score(self, x, y):
        """Return the coefficient of determination of the model fit"""
        raise NotImplementedError

class GaussianVAE(VAE):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.log_lambda_mean = _S(_Z(1))
        log_lambda_log_prec = _S(_Z(1))
        self.log_lambda_prec = 1e-3 + T.nnet.softplus(log_lambda_log_prec)
        super().__init__(X, y, a,
                         hyperparam_means=[self.log_lambda_mean],
                         hyperparam_log_precs=[log_lambda_log_prec],
                         **kwargs)
        self.lambda_ = self.min_prec + T.nnet.softplus(self.log_lambda_mean)
        self.predict = _F(inputs=[self.X], outputs=self.eta_mean, allow_input_downcast=True)
        R = 1 - T.sqr(self.y - self.eta_mean).sum() / T.sqr(self.y - self.y.mean()).sum()
        self.score = _F(inputs=[self.X, self.y], outputs=R, allow_input_downcast=True)
        self.loss = _F(inputs=[self.X, self.y], outputs=T.sqr(self.y - self.eta_mean).sum(), allow_input_downcast=True)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, theta_0)] assuming a linear link."""
        phi = T.exp(T.addbroadcast(self.min_prec + T.nnet.softplus(self.log_lambda_mean + T.sqrt(1 / self.log_lambda_prec) * phi_raw), 1))
        F = -.5 * (-T.log(phi) + T.sqr(y - eta) * phi)
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
