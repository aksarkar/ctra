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
    def __init__(self, X_, y_, a_, stoch_samples=500, learning_rate=1e-4,
                 warmup_rate=1e-3, hyperparam_means=None,
                 hyperparam_logit_precs=None, random_state=None, **kwargs):
        """Compile the Theano function which takes a gradient step"""
        super().__init__(X_, y_, a_, None)
        self.warmup_rate = warmup_rate
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
        q_logit_pi_logit_prec = _S(_Z(m), name='q_logit_pi_logit_prec')
        q_log_tau_mean = _S(_Z(m), name='q_log_tau_mean')
        q_log_tau_logit_prec = _S(_Z(m), name='q_log_tau_logit_prec')

        # Variational parameters
        n, p = X_.shape
        q_logit_z = _S(_Z(p), name='q_logit_z')
        q_z = T.nnet.sigmoid(q_logit_z)
        q_theta_mean = _S(_Z(p), name='q_theta_mean')
        q_theta_logit_prec = _S(_Z(p), name='q_theta_logit_prec')
        q_theta_prec = self.max_prec * T.nnet.sigmoid(q_theta_logit_prec)
        self.params = [q_logit_z, q_theta_mean, q_theta_logit_prec]

        # These will include model-specific terms. Assume everything is
        # Gaussian on the variational side to simplify
        self.hyperparam_means = [q_logit_pi_mean, q_log_tau_mean]
        self.hyperparam_logit_precs = [q_logit_pi_logit_prec, q_log_tau_logit_prec]
        if hyperparam_means is not None:
            self.hyperparam_means.extend(hyperparam_means)
        if hyperparam_logit_precs is not None:
            self.hyperparam_logit_precs.extend(hyperparam_logit_precs)

        self.hyperprior_means = [numpy.repeat(-2, m).astype(_real), _Z(m)]
        self.hyperprior_logit_precs = [_Z(m), _Z(m)]
        for _ in hyperparam_means:
            self.hyperprior_means.append(_Z(m))
            self.hyperprior_logit_precs.append(_Z(m))
        self.hyperprior_max_prec = 100

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
        phi = [T.addbroadcast(mean + T.sqrt(1 / (self.hyperprior_max_prec * T.nnet.sigmoid(logit_prec))) * phi_raw, 1).dimshuffle(0, 'x')
               for mean, logit_prec in zip(self.hyperparam_means, self.hyperparam_logit_precs)]

        # We need to warm up the objective function in order to avoid
        # degenerate solutions where all variational free parameters go to
        # zero. The idea is given in Sønderby et al., NIPS 2016
        temperature = T.clip(T.cast(warmup_rate * epoch, _real), 0, 1)
        error = self._llik(y, eta, phi_raw)
        pi = T.addbroadcast(T.nnet.sigmoid(q_logit_pi_mean), 0)
        tau = T.addbroadcast(T.nnet.softplus(q_log_tau_mean), 0)
        kl = (.5 * T.sum(q_z * (1 + tau - T.log(q_theta_prec) -
                                tau * (T.sqr(q_theta_mean) + 1 / q_theta_prec)))
              - T.sum(q_z * T.log(q_z / pi) + (1 - q_z) * T.log((1 - q_z) / (1 - pi))))
        kl_hyper = 0
        for mean, logit_prec, prior_mean, prior_logit_prec in zip(self.hyperparam_means, self.hyperparam_logit_precs, self.hyperprior_means, self.hyperprior_logit_precs):
            prec = self.max_prec * T.nnet.sigmoid(logit_prec)
            prior_prec = self.hyperprior_max_prec * T.nnet.sigmoid(prior_logit_prec)
            kl_hyper += .5 * T.sum(1 + T.log(prior_prec) - T.log(prec) + prior_prec * (T.sqr(mean - prior_mean) + 1 / prec))
        elbo = error - temperature * (kl + kl_hyper)

        logger.debug('Compiling the Theano functions')
        init_updates = [(param, _Z(p)) for param in self.params]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_means, self.hyperprior_means)]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_logit_precs, self.hyperprior_logit_precs)]
        self.initialize = _F(inputs=[], outputs=[], updates=init_updates)

        sgd_updates = [(param, param + numpy.array(learning_rate, dtype=_real) * g)
                       for param, g in zip(self.params, T.grad(elbo, self.params))]
        sgd_updates += [(param, param + numpy.array(learning_rate, dtype=_real) * g)
                        for param, g in zip(self.hyperparam_means + self.hyperparam_logit_precs,
                                            T.grad(elbo, self.hyperparam_means + self.hyperparam_logit_precs))]
        sgd_givens = {X: X_.astype(_real), y: y_.astype(_real), a: a_}
        self.sgd_step = _F(inputs=[epoch], outputs=[error, kl + kl_hyper], updates=sgd_updates, givens=sgd_givens)
        trace_outputs = ([epoch, error, kl, kl_hyper, q_theta_prec.max(), q_z.max()] + self.hyperparam_means)
        self.trace = _F(inputs=[epoch], outputs=trace_outputs, givens=sgd_givens)
        opt_outputs = [elbo, q_z, q_theta_mean, T.nnet.sigmoid(q_logit_pi_mean), 1 / T.nnet.sigmoid(q_logit_pi_logit_prec),
                       T.nnet.softplus(q_log_tau_mean), 1 / T.nnet.sigmoid(q_log_tau_logit_prec)]
        self.opt = _F(inputs=[epoch], outputs=opt_outputs, givens=sgd_givens)
        logger.debug('Finished initializing')

        test_point = T.fvector()
        query_givens = sgd_givens
        query_givens[epoch] = numpy.array(0, dtype='int32')
        query_givens[q_logit_z] = test_point
        query = _F(inputs=[test_point], outputs=T.grad(error, q_logit_z), givens=query_givens)
        grid = numpy.mgrid[-10:10:1, -10:10:1].astype(_real)
        z = numpy.array([query(x) for x in numpy.rollaxis(grid, 0, 3).reshape(-1, 2)])
        z = numpy.rollaxis(z.reshape(20, 20, 2), 2)
        figure()
        quiver(grid[0], grid[1], z[0], z[1], pivot='mid')
        xlabel('logit(alpha0)')
        ylabel('logit(alpha1)')
        savefig('test.pdf')
        sys.exit(0)

    def _llik(self, *args):
        raise NotImplementedError

    def log_weight(self, *args):
        raise NotImplementedError
        
    def fit(self, atol=1e-3, **kwargs):
        self.initialize()
        t = 0
        logger.debug(' '.join('{:.3g}'.format(numpy.asscalar(x)) for x in self.trace(t)))
        error_ = float('-inf')
        kl_ = float('-inf')
        while t < 1000:
            t += 1
            error, kl = self.sgd_step(epoch=t)
            assert numpy.isfinite(kl) and numpy.isfinite(error)
            if not t % 50:
                logger.debug('\t'.join(['epoch', 'error', 'kl', 'klh', 'vars', 'pip', 'pi', 'tau', 'sigma2']))
            logger.debug('\t'.join('{:.5g}'.format(numpy.asscalar(x)) for x in self.trace(t)))
            if t > 1 / self.warmup_rate and (error - kl) < (error_ - kl_):
                logger.debug('ELBO decreased')
                break
            elif numpy.isclose(kl, kl_, atol=atol) and numpy.isclose(error, error_, atol=atol):
                break
            else:
                error_, kl_ = error, kl
        logger.info('Converged at epoch {}'.format(t))
        self._evidence, self.pip, self.q_theta_mean, self.pi, self.logit_pi_var, self.tau, self.log_tau_var = self.opt(epoch=t)
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
        log_sigma2_logit_prec = _S(_Z(1))
        self.log_sigma2_prec = 1000 * T.nnet.sigmoid(log_sigma2_logit_prec)
        super().__init__(X, y, a,
                         hyperparam_means=[self.log_sigma2_mean],
                         hyperparam_logit_precs=[log_sigma2_logit_prec],
                         **kwargs)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, theta_0)] assuming a linear link."""
        phi = T.addbroadcast(T.nnet.softplus(self.log_sigma2_mean + T.sqrt(1 / self.log_sigma2_prec) * phi_raw), 1)
        F = -.5 * (T.log(phi) + T.sqr(y - eta) / phi)
        return T.mean(T.sum(F, axis=1))

class LogisticVAE(VAE):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.bias_mean = _S(_Z(1))
        bias_logit_prec = _S(_Z(1))
        self.bias_prec = 1000 * T.nnet.sigmoid(bias_logit_prec)
        super().__init__(X, y, a, hyperparam_means=[self.bias_mean],
                         hyperparam_logit_precs=[bias_logit_prec],
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
