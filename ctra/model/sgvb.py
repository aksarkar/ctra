"""Maximize evidence lower bound as proxy for marginal likelihood using
Stochastic Gradient Variational Bayes

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import logging

import numpy
import scipy.misc
import scipy.special
import theano
import theano.tensor as T

logger = logging.getLogger(__name__)

_real = theano.config.floatX
_F = theano.function
_Z = lambda n: numpy.zeros(n).astype(_real)

def _S(x, **kwargs):
    return theano.shared(x, borrow=True, **kwargs)

def kl_normal_normal(mean, prec, prior_mean, prior_prec):
    return .5 * (1 - T.log(prior_prec) + T.log(prec) + prior_prec * (T.sqr(mean - prior_mean) + 1 / prec))

def rmsprop(loss, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """RMSProp updates (from Lasagne)

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}
    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """
    grads = theano.grad(loss, params)
    updates = collections.OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(_Z(value.shape).astype(_real),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates

class SGVB:
    """Base class providing the generic implementation. Specialized sub-classes are
needed for specific likelihoods.

    """
    def __init__(self, X_, y_, a_, stoch_samples=50, learning_rate=0.1,
                 hyperparam_means=None, hyperparam_log_precs=None,
                 random_state=None, minibatch_n=None, **kwargs):
        logger.debug('Building the Theano graph')
        # Observed data. This needs to be symbolic for minibatches
        # TODO: borrow, GPU transfer, HDF5 transfer
        self.X_ = _S(X_.astype(_real))
        self.y_ = _S(y_.astype(_real))
        self.p = numpy.array(list(collections.Counter(a_).values()), dtype='int')

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
        self.q_w_mean = _S(_Z(m), name='q_w_mean')
        self.q_w_log_prec = _S(_Z(m), name='q_w_log_prec')
        self.q_b_mean = _S(_Z(1), name='q_b_mean')
        self.q_b_log_prec = _S(_Z(1), name='q_b_log_prec')
        self.q_log_tau0_mean = _S(_Z(1), name='q_log_tau0_mean')
        self.q_log_tau0_log_prec = _S(_Z(1), name='q_log_tau0_log_prec')
        self.q_tau0_mean = T.exp(self.q_log_tau0_mean)
        self.q_log_tau1_mean = _S(_Z(m), name='q_log_tau1_mean')
        self.q_log_tau1_log_prec = _S(_Z(m), name='q_log_tau1_log_prec')
        self.q_tau1_mean = T.exp(T.addbroadcast(self.q_log_tau0_mean, 0) - self.q_log_tau1_mean)

        # We don't need to use the hyperparameter noise samples for these
        # parameters because we can deal with them analytically
        pi = T.nnet.sigmoid(T.dot(self.A, self.q_w_mean) + T.addbroadcast(self.q_b_mean, 0))
        # Share spike variance across all groups
        tau0 = T.addbroadcast(self.q_tau0_mean, 0)
        tau1 = T.dot(self.A, self.q_tau1_mean)

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
        self.hyperparam_means = [self.q_w_mean, self.q_b_mean, self.q_log_tau0_mean, self.q_log_tau1_mean]
        self.hyperparam_log_precs = [self.q_w_log_prec, self.q_b_log_prec, self.q_log_tau0_log_prec, self.q_log_tau1_log_prec]
        if hyperparam_means is not None:
            self.hyperparam_means.extend(hyperparam_means)
        if hyperparam_log_precs is not None:
            self.hyperparam_log_precs.extend(hyperparam_log_precs)

        self.hyperprior_means = [_Z(m), _Z(1), _Z(1), _Z(m)]
        self.hyperprior_log_precs = [_Z(m), _Z(1), _Z(1), _Z(m)]
        if hyperparam_means is None:
            hyperparam_means = []
        for _ in hyperparam_means:
            self.hyperprior_means.append(_Z(1))
            self.hyperprior_log_precs.append(_Z(1))

        # We need to perform inference on minibatches of samples for speed. Rather
        # than taking balanced subsamples, we take a sliding window over a
        # permutation which is balanced in expectation.
        epoch = T.iscalar(name='epoch')

        # Pre-generate stochastic samples
        if random_state is None:
            self._R = numpy.random
        else:
            self._R = random_state
        noise = _S(self._R.normal(size=(5 * stoch_samples, minibatch_n)).astype(_real), name='noise')

        # Re-parameterize eta = X theta (Kingma, Salimans, & Welling NIPS
        # 2015), and backpropagate through the RNG (Kingma & Welling, ICLR
        # 2014).
        self.eta_mean = T.dot(self.X, self.q_z * self.q_theta_mean)
        eta_var = T.dot(T.sqr(self.X), self.q_z / self.q_theta_prec + self.q_z * (1 - self.q_z) * T.sqr(self.q_theta_mean))
        eta_minibatch = epoch % 5
        eta_raw = noise[eta_minibatch * stoch_samples:(eta_minibatch + 1) * stoch_samples]
        eta = self.eta_mean + T.sqrt(eta_var) * eta_raw

        # We need to generate independent noise samples for model parameters
        # besides the GSS parameters/hyperparameters (biases, variances in
        # likelihood)
        phi_minibatch = (epoch + 1) % 5
        phi_raw = noise[phi_minibatch * stoch_samples:(phi_minibatch + 1) * stoch_samples,:1]

        error = self._llik(self.y, eta, phi_raw)
        # Rasmussen and Williams, Eq. A.23, conditioning on q_z (alpha in our
        # notation)
        zero = T.addbroadcast(T.constant(_Z(1)), 0)
        kl_qtheta_ptheta = (self.q_z * kl_normal_normal(self.q_theta_mean, self.q_theta_prec, zero, tau1) +
                            (1 - self.q_z) * kl_normal_normal(self.q_theta_mean, self.q_theta_prec, zero, tau0)).sum()
        # Rasmussen and Williams, Eq. A.22
        kl_qz_pz = T.sum(self.q_z * T.log(self.q_z / pi) + (1 - self.q_z) * T.log((1 - self.q_z) / (1 - pi)))
        kl_hyper = 0
        for mean, log_prec, prior_mean, prior_log_prec in zip(self.hyperparam_means, self.hyperparam_log_precs, self.hyperprior_means, self.hyperprior_log_precs):
            prec = self.min_prec + T.nnet.softplus(log_prec)
            prior_prec = self.min_prec + T.nnet.softplus(prior_log_prec)
            kl_hyper += kl_normal_normal(mean, prec, prior_mean, prior_prec).sum()
        kl = kl_qtheta_ptheta + kl_qz_pz + kl_hyper
        # Kingma & Welling 2013 (eq. 8)
        elbo = (error - kl) * self.scale_n

        logger.debug('Compiling the Theano functions')
        init_updates = [(param, self._R.normal(size=p).astype(_real)) for param in self.params]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_means, self.hyperprior_means)]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_log_precs, self.hyperprior_log_precs)]
        self.initialize = _F(inputs=[], outputs=[], updates=init_updates)

        self.variational_params = self.params + self.hyperparam_means + self.hyperparam_log_precs
        # Lasagne minimizes, so flip the sign
        sgd_updates = rmsprop(-elbo, self.variational_params, learning_rate=learning_rate)
        sample_minibatch = epoch % (n // minibatch_n)
        sgd_givens = {self.X: self.X_[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n],
                      self.y: self.y_[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]}
        self.sgd_step = _F(inputs=[epoch], outputs=elbo, updates=sgd_updates, givens=sgd_givens)
        self._trace = _F(inputs=[epoch],
                         outputs=[epoch, elbo, error, kl_qz_pz, kl_qtheta_ptheta, kl_hyper] +
                         self.variational_params, givens=sgd_givens)
        opt_outputs = [elbo, self.q_z, self.q_theta_mean, 1 / self.q_theta_prec,
                       T.nnet.sigmoid(self.q_w_mean + T.addbroadcast(self.q_b_mean, 0))]
        self.opt = _F(inputs=[], outputs=opt_outputs,
                      givens=[(phi_raw, numpy.zeros((1, 1), dtype=_real)),
                              (eta_raw, numpy.zeros((1, n), dtype=_real)),
                              (self.X, self.X_),
                              (self.y, self.y_)])

        logger.debug('Finished initializing')

    def loss(self, *args):
        """THEANO FUNCTION"""
        raise NotImplementedError

    def _llik(self, *args):
        """Return the log likelihood of the (mini)batch"""
        raise NotImplementedError

    def fit(self, loc=0, max_epochs=100, xv=None, yv=None, trace=False, **kwargs):
        logger.debug('Starting SGD')
        self.initialize()
        self.q_logit_z.set_value(self._R.normal(loc=loc, size=self.q_logit_z.get_value().shape).astype(_real))
        t = 0
        elbo_ = float('-inf')
        loss = float('inf')
        self.trace = []
        while t < max_epochs * self.scale_n:
            t += 1
            elbo = self.sgd_step(epoch=t)
            if not t % (10 * self.scale_n):
                elbo_ = elbo
                if elbo < elbo_:
                    logger.warn('ELBO increased, stopping early')
                    break
                self.validation_loss = self.loss(xv, yv)
                loss = self.validation_loss
                outputs = self._trace(t)[:6]
                outputs.append(self.score(self.X_.get_value(), self.y_.get_value()))
                outputs.append(self.score(xv, yv))
                if not self.trace or trace:
                    self.trace.append(outputs)
                else:
                    self.trace[0] = outputs
                logger.debug('\t'.join('{:.3g}'.format(numpy.asscalar(x)) for x in outputs))
            if not numpy.isfinite(elbo):
                logger.warn('ELBO infinite. Stopping early')
                break
        self._evidence, self.pip, self.theta, self.theta_var, self.pi = self.opt()
        logger.info('Converged at epoch {}'.format(t))
        return self

    def predict(self, x):
        """THEANO FUNCTION"""
        raise NotImplementedError

    def score(self, x, y):
        """THEANO FUNCTION"""
        raise NotImplementedError

class GaussianSGVB(SGVB):
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
        self.predict = _F(inputs=[self.X], outputs=self.eta_mean, allow_input_downcast=True)
        # Coefficient of determination
        R = 1 - T.sqr(self.y - self.eta_mean).sum() / T.sqr(self.y - self.y.mean()).sum()
        self.score = _F(inputs=[self.X, self.y], outputs=R, allow_input_downcast=True)
        self.loss = _F(inputs=[self.X, self.y], outputs=T.sqr(self.y - self.eta_mean).sum(), allow_input_downcast=True)

    def _llik(self, y, eta, phi_raw):
        phi = T.exp(T.addbroadcast(self.min_prec + T.nnet.softplus(self.log_lambda_mean + T.sqrt(1 / self.log_lambda_prec) * phi_raw), 1))
        F = -.5 * (-T.log(phi) + T.sqr(y - eta) * phi)
        return T.mean(T.sum(F, axis=1))

class LogisticSGVB(SGVB):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.bias_mean = _S(_Z(1))
        bias_log_prec = _S(_Z(1))
        self.bias_prec = T.nnet.softplus(bias_log_prec)
        super().__init__(X, y, a, hyperparam_means=[self.bias_mean],
                         hyperparam_log_precs=[bias_log_prec],
                         **kwargs)

        eta = self.eta_mean + T.addbroadcast(self.bias_mean, 0)
        log_loss = -(self.y * eta - T.nnet.softplus(eta)).sum()
        self.loss = _F(inputs=[self.X, self.y], outputs=log_loss, allow_input_downcast=True)

        # F measure
        yhat = T.cast(T.nnet.sigmoid(eta) > 0.5, 'int8')
        precision = (yhat * self.y).sum() / yhat.sum()
        recall = (yhat * self.y).sum() / self.y.sum()
        F = 2 * precision * recall / (precision + recall)
        self.score = _F(inputs=[self.X, self.y], outputs=F, allow_input_downcast=True)
        self.predict = _F(inputs=[self.X], outputs=yhat)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, ...)] assuming a logit link.

        Fit an intercept phi using SGVB, assuming p(phi) ~ N(0, 1), q(phi) ~ N(m, v).

        """
        phi = T.addbroadcast(T.nnet.softplus(self.bias_mean + T.sqrt(1 / self.bias_prec) * phi_raw), 1)
        F = y * (eta + phi) - T.nnet.softplus(eta + phi)
        return T.mean(T.sum(F, axis=1))

class ProbitSGVB(SGVB):
    def __init__(self, X, y, a, **kwargs):
        super().__init__(X, y, a, **kwargs)
        self.predict = _F(inputs=[self.X], outputs=[])

        eta = self.eta_mean
        log_loss = -(self.y * eta - T.nnet.softplus(eta)).sum()
        self.loss = _F(inputs=[self.X, self.y], outputs=log_loss, allow_input_downcast=True)

        # F measure
        yhat = T.cast(T.sqrt(2) * T.erfinv(2 * T.dot(self.X, self.theta) - 1) > 0.5, 'int8')
        precision = (yhat * self.y).sum() / yhat.sum()
        recall = (yhat * self.y).sum() / self.y.sum()
        F = 2 * precision * recall / (precision + recall)
        self.score = _F(inputs=[self.X, self.y], outputs=F, allow_input_downcast=True)
        self.predict = _F(inputs=[self.X], outputs=yhat)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, theta_0)] assuming a probit link.

        Fix (latent) residual precision to 1.

        """
        F = .5 + .5 * T.erf(eta / T.sqrt(2))
        return T.mean(T.sum(F, axis=1))
