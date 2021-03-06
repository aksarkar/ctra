"""Maximize evidence lower bound using Stochastic Gradient Variational Bayes

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import logging

from matplotlib.pyplot import *
import numpy
import scipy.misc
import scipy.special
import theano
import theano.tensor as T

logger = logging.getLogger(__name__)

_real = theano.config.floatX
_F = theano.function
_Z = lambda n: numpy.zeros(n).astype(_real)
_O = lambda n: numpy.ones(n).astype(_real)

def _S(x, **kwargs):
    return theano.shared(x, borrow=True, **kwargs)

def kl_normal_normal(mean, prec, prior_mean, prior_prec):
    return .5 * (1 - T.log(prior_prec) + T.log(prec) + prior_prec * (T.sqr(mean - prior_mean) + 1 / prec))

def rmsprop(loss, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, **kwargs):
    """RMSProp (from Lasagne)

    Tieleman & Hinton 2012

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

def clipped_sigmoid(x):
    lim = numpy.log(numpy.finfo('float32').resolution)
    return T.nnet.sigmoid(T.clip(x, lim, -lim))

class SGVB:
    """Base class providing the generic implementation. Specialized sub-classes are
needed for specific likelihoods.

    """
    def __init__(self, X_, y_, a_, m0=None, stoch_samples=50,
                 learning_rate=0.1, minibatch_n=None, rho=0.9, weights=None,
                 hyperparam_means=None, hyperparam_log_precs=None,
                 random_state=None, prior_mean_b=None, prior_mean_c=None):
        """Initialize the model

        X_ - n x p dosages
        y_ - n x 1 phenotypes
        a_ - p x 1 annotations (entries in {1..m})
        m0 - Genome-wide model (needed for baseline b, c)
        stoch_samples - Noise samples for SGVB
        learning_rate - initial learning rate for RMSprop
        minibatch_n - Minibatch size
        rho - decay rate for RMSprop
        weights - sample weights (use to efficiently represent bootstraps)
        hyperparam_means - Means of variational surrogates for model-specific parameters
        hyperparam_log_precs - Log precisions of variational surrogates for model-specific parameters
        random_state - numpy RandomState

        """
        logger.debug('Building the Theano graph')
        # Observed data. This needs to be symbolic for minibatches
        n, p = X_.shape
        self.X_ = _S(X_.astype(_real))
        self.y_ = _S(y_.astype(_real))
        if weights is None:
            weights = numpy.ones(n)
        self.w_ = _S(weights.astype('int32'))

        if a_.ndim == 1:
            # One-hot encode the annotations
            m = len(collections.Counter(a_).values())
            A = numpy.zeros((p, m)).astype('i1')
            A[range(p), a_] = 1
        else:
            m = a_.shape[1]
            A = a_
        self.A = _S(A)

        self.X = T.fmatrix(name='X')
        self.y = T.fvector(name='y')
        self.w = T.ivector(name='w')

        if minibatch_n is None:
            minibatch_n = n
        self.scale_n = n / minibatch_n

        # Variational surrogate for target hyperposterior
        # p(theta_j) = pi_j N(0, tau_j^-1) +
        #              (1 - pi_j) \delta
        # logit(pi_j) = A_j w + b
        self.q_w_mean = _S(_Z(m), name='q_w_mean')
        self.q_w_log_prec = _S(_Z(m), name='q_w_log_prec')
        self.q_b_mean = _S(_Z(1), name='q_b_mean')
        self.q_b_log_prec = _S(_Z(1), name='q_b_log_prec')
        # tau_j = eps + softplus(A_j v + c)
        self.min_prec = 1e-3
        self.q_v_mean = _S(_Z(m), name='q_v_mean')
        self.q_v_log_prec = _S(_Z(m), name='q_v_log_prec')
        self.q_c_mean = _S(_Z(1), name='q_c_mean')
        self.q_c_log_prec = _S(_Z(1), name='q_c_log_prec')

        # We don't need to use the hyperparameter noise samples for these
        # parameters because we can deal with them analytically
        if m0 is None:
            pi = clipped_sigmoid(T.addbroadcast(self.q_b_mean, 0))
            tau = self.min_prec + T.nnet.softplus(T.addbroadcast(self.q_c_mean, 0))
        else:
            self.b = m0.b
            self.c = m0.c
            pi = clipped_sigmoid(T.dot(self.A, self.q_w_mean) + self.b)
            tau = self.min_prec + T.nnet.softplus(T.dot(self.A, self.q_v_mean) + self.c)

        # Variational parameters
        self.q_logit_z = _S(_Z(p), name='q_logit_z')
        self.q_z = clipped_sigmoid(self.q_logit_z)
        self.q_theta_mean = _S(_Z(p), name='q_theta_mean')
        self.q_theta_log_prec = _S(_Z(p), name='q_theta_log_prec')
        self.q_theta_prec = self.min_prec + T.nnet.softplus(self.q_theta_log_prec)
        self.params = [self.q_logit_z, self.q_theta_mean, self.q_theta_log_prec]

        if m0 is None:
            self.hyperparam_means = [self.q_b_mean, self.q_c_mean]
            self.hyperparam_log_precs = [self.q_b_log_prec, self.q_c_log_prec]
            # Default priors
            self.hyperprior_means = [numpy.array([-numpy.log(p)], dtype=_real), _Z(1)]
            if prior_mean_b is not None:
                self.hyperprior_means[0] = numpy.array([prior_mean_b], dtype=_real)
            if prior_mean_c is not None:
                self.hyperprior_means[1] = numpy.array([prior_mean_c], dtype=_real)
            self.hyperprior_precs = [_O(1), _O(1)]
        else:
            self.hyperparam_means = [self.q_w_mean, self.q_v_mean]
            self.hyperparam_log_precs = [self.q_w_log_prec, self.q_v_log_prec]
            self.hyperprior_means = [_Z(m), _Z(m)]
            self.hyperprior_precs = [_O(m), _O(m)]

        if hyperparam_means is not None:
            # Include model-specific terms. Assume everything is Gaussian on
            # the variational side to simplify
            self.hyperparam_means.extend(hyperparam_means)
        else:
            hyperparam_means = []
        if hyperparam_log_precs is not None:
            self.hyperparam_log_precs.extend(hyperparam_log_precs)
        for _ in hyperparam_means:
            self.hyperprior_means.append(_Z(1))
            self.hyperprior_precs.append(_O(1))

        # We need to perform inference on minibatches of samples for speed. Rather
        # than taking balanced subsamples, we take a sliding window over a
        # permutation which is balanced in expectation.
        epoch = T.iscalar(name='epoch')

        # Pre-generate stochastic samples
        stoch_sample_batches = 10
        if random_state is None:
            self._R = numpy.random
        else:
            self._R = random_state
        noise = _S(self._R.normal(size=(stoch_sample_batches * stoch_samples, minibatch_n)).astype(_real), name='noise')

        # Local reparameterization eta = X theta (Kingma, Salimans, & Welling
        # NIPS 2015)
        self.theta_posterior_mean = self.q_z * self.q_theta_mean
        self.theta_posterior_var = self.q_z / self.q_theta_prec + self.q_z * (1 - self.q_z) * T.sqr(self.q_theta_mean)
        self.eta_mean = T.dot(self.X, self.theta_posterior_mean)
        eta_var = T.dot(T.sqr(self.X), self.theta_posterior_var)
        eta_minibatch = epoch % stoch_sample_batches
        eta_raw = noise[eta_minibatch * stoch_samples:(eta_minibatch + 1) * stoch_samples]
        eta = self.eta_mean + T.sqrt(eta_var) * eta_raw

        # We need to generate independent noise samples for model parameters
        # besides the GSS parameters/hyperparameters (biases, variances in
        # likelihood)
        phi_minibatch = (epoch + 1) % stoch_sample_batches
        phi_raw = noise[phi_minibatch * stoch_samples:(phi_minibatch + 1) * stoch_samples,:len(hyperparam_means)]

        error = self._llik(self.y, eta, phi_raw)
        # Rasmussen and Williams, Eq. A.23, conditioning on q_z (alpha in our
        # notation)
        zero = T.addbroadcast(T.constant(_Z(1)), 0)
        kl_qtheta_ptheta = (self.q_z * kl_normal_normal(self.q_theta_mean, self.q_theta_prec, zero, tau)).sum()
        # Rasmussen and Williams, Eq. A.22
        kl_qz_pz = T.sum(self.q_z * T.log(self.q_z / pi) + (1 - self.q_z) * T.log((1 - self.q_z) / (1 - pi)))
        kl_hyper = 0
        for mean, log_prec, prior_mean, prior_prec in zip(self.hyperparam_means, self.hyperparam_log_precs, self.hyperprior_means, self.hyperprior_precs):
            kl_hyper += kl_normal_normal(mean, self.min_prec + T.nnet.softplus(log_prec), prior_mean, prior_prec).sum()
        # Kingma & Welling 2013 (eq. 24)
        elbo = self.scale_n * (error - kl_qtheta_ptheta - kl_qz_pz) - kl_hyper

        logger.debug('Compiling the Theano functions')
        init_updates = [(param, self._R.normal(scale=0.1, size=p).astype(_real)) for param in self.params]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_means, self.hyperprior_means)]
        init_updates += [(param, val) for param, val in zip(self.hyperparam_log_precs, self.hyperprior_precs)]
        self.initialize = _F(inputs=[], outputs=[], updates=init_updates)

        self.variational_params = self.params + self.hyperparam_means + self.hyperparam_log_precs
        sgd_updates = rmsprop(-elbo, self.variational_params, learning_rate=learning_rate, rho=rho)
        sample_minibatch = epoch % (n // minibatch_n)
        sgd_givens = {self.X: self.X_[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n],
                      self.y: self.y_[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n],
                      self.w: self.w_[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]}
        self.sgd_step = _F(inputs=[epoch], outputs=elbo, updates=sgd_updates, givens=sgd_givens)

        self._trace = _F(inputs=[epoch],
                         outputs=[epoch, elbo, error, kl_qz_pz, kl_qtheta_ptheta, kl_hyper] +
                         self.variational_params, givens=sgd_givens)
        opt_outputs = [elbo, self.q_z, self.theta_posterior_mean, self.theta_posterior_var, self.q_w_mean, self.q_b_mean,
                       self.q_v_mean, self.q_c_mean]
        opt_givens = [(phi_raw, numpy.zeros((1, len(hyperparam_means)), dtype=_real)),
                      (eta_raw, numpy.zeros((1, n), dtype=_real)),
                      (self.X, self.X_),
                      (self.y, self.y_),
                      (self.w, self.w_)]
        self.opt = _F(inputs=[], outputs=opt_outputs, givens=opt_givens)

        self._trace = _F(inputs=[epoch],
                         outputs=[epoch, elbo, error, kl_qz_pz, kl_qtheta_ptheta, kl_hyper] +
                         self.variational_params, givens=sgd_givens)
        self._trace_stoch_grad = _F(inputs=[epoch], outputs=theano.grad(elbo, self.variational_params), givens=sgd_givens)
        self._trace_full_grad = _F(inputs=[], outputs=theano.grad(elbo, self.variational_params), givens=opt_givens)

        logger.debug('Finished initializing')

    def loss(self, *args):
        """THEANO FUNCTION"""
        raise NotImplementedError

    def _llik(self, *args):
        """Return the log likelihood of the (mini)batch"""
        raise NotImplementedError

    def fit(self, xv, yv, max_epochs=20):
        """Fit the model

        xv - hold out validation predictors (for tracing learning)
        yv - hold out validation responses (for tracing learning)
        max_epochs - maximum number of full data passes

        """
        self.initialize()
        logger.debug('Starting SGD')
        t = 0
        elbo_ = float('-inf')
        loss = float('inf')
        self.grads = []
        while t < max_epochs * self.scale_n:
            t += 1
            elbo = self.sgd_step(epoch=t)
            if not t % self.scale_n:
                self.grads.append([self._trace_stoch_grad(t), self._trace_full_grad()])
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
                logger.debug('\t'.join('{:.3g}'.format(numpy.asscalar(x)) for x in outputs))
            if not numpy.isfinite(elbo):
                logger.warn('ELBO infinite. Stopping early')
                break
        self.validation_loss = self.loss(xv, yv)
        self._evidence, self.pip, self.theta, self.theta_var, self.w, self.b, self.v, self.c = self.opt()
        return self

    def predict(self, x):
        """THEANO FUNCTION"""
        raise NotImplementedError

    def score(self, x, y):
        """THEANO FUNCTION"""
        raise NotImplementedError

    def plot(self, s):
        q = numpy.logical_or(self.pip > 0.1, s.theta != 0)
        nq = numpy.count_nonzero(q)
        fig, ax = subplots(4, 1)
        fig.set_size_inches(6, 8)
        xlabel('True and false positive variants')
        ax[0].bar(range(nq), s.maf[q])
        ax[0].set_ylabel('MAF')
        ax[1].bar(range(nq), s.theta[q])
        ax[1].set_ylabel('True effect size')
        ax[2].bar(range(nq), self.theta[q])
        ax[2].set_ylabel('Estimated effect size')
        ax[3].bar(range(nq), self.pip[q])
        ax[3].set_ylabel('PIP')
        return fig

    def plot_neighborhood(self):
        """Plot random 2d-slices around the current point

        http://stanford.edu/class/ee364a/lectures/functions.pdf

        """
        _, _, _, _, _, _, *loc = self._trace(0)
        query = numpy.linspace(-2, 2, 100)
        figure()
        for i in range(10):
            direction = [numpy.random.normal(size=p.shape) for p in loc]
            vals = []
            for t in query:
                for p, v, d in zip(self.params, loc, direction):
                    p.set_value(numpy.array(v + t * d, dtype='float32'))
                vals.append(self.opt()[0])
            plot(query, vals)
        axvline()
        savefig('diagnostic.pdf')
        close()
        for p, v in zip(self.params, loc):
            p.set_value(numpy.array(v, dtype='float32'))

    def plot_fdr(self, s):
        false_pos = [(P, N) for N, (P, I) in
                     enumerate(sorted(zip(self.pip, s.theta != 0), reverse=True))
                     if not I]
        fdr = numpy.array([(P, (I + 1) / N) for I, (P, N) in enumerate(fps)])
        gcf()
        clf()
        plot(fdr)
        axhline(intercept=0.05)
        xlabel('PIP')
        ylabel('FDR')
        savefig('fdr.pdf')
        close()

class GaussianSGVB(SGVB):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.log_lambda_mean = _S(_Z(1), name='log_lambda_mean')
        log_lambda_log_prec = _S(_Z(1), name='log_lambda_log_prec')
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
        # [stoch_samples, 1]
        phi = T.addbroadcast(self.min_prec + T.nnet.softplus(self.log_lambda_mean + T.sqrt(1 / self.log_lambda_prec) * phi_raw), 1)
        # [stoch_samples, n] * [n, 1]
        F = -.5 * T.dot((-T.log(phi) + T.sqr(y - eta) * phi), self.w)
        return T.mean(F)

class LogisticSGVB(SGVB):
    def __init__(self, X, y, a, **kwargs):
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        self.bias_mean = _S(_Z(1), name='bias_mean')
        bias_log_prec = _S(_Z(1), name='bias_log_prec')
        self.bias_prec = T.nnet.softplus(bias_log_prec)
        super().__init__(X, y, a, **kwargs)

        prob_y = clipped_sigmoid(self.eta_mean)
        self.predict_proba = _F(inputs=[self.X], outputs=prob_y)
        # Brier score loss
        self.loss = _F(inputs=[self.X, self.y], outputs=T.sqr(self.y - prob_y).sum(), allow_input_downcast=True)

        # GLM coefficient of determination
        R = 1 - T.sqr(self.y - prob_y).sum() / T.sqr(self.y - self.y.mean()).sum()
        self.score = _F(inputs=[self.X, self.y], outputs=R, allow_input_downcast=True)
        yhat = T.cast(prob_y > 0.5, 'int8')
        self.predict = _F(inputs=[self.X], outputs=yhat)

    def _llik(self, y, eta, phi_raw):
        """Return E_q[ln p(y | eta, ...)] assuming a logit link.

        Fit an intercept phi using SGVB, assuming p(phi) ~ N(0, 1), q(phi) ~ N(m, v).

        """
        F = y * eta - T.nnet.softplus(eta)
        return T.mean(T.sum(F, axis=1))

class VBSLMM(SGVB):
    def __init__(self, X, y, a, **kwargs):
        """Optimize the variational objective with respect to the BSLMM likelihood
(Zhou et al PLoS Genet 2013), assuming the group spike and slab prior on the
coefficients.

        y \sim N(X \theta, \lambda_1^-1 K + \lambda_2^-1 I)

        Use the FastLMM trick to simplify this:

        X = U D V'
        K = U D^2 U'
        y \sim N(X \theta, U (\lambda_1^-1 D^2 + \lambda_2^-1 I) U')
        U' y \sim N(U' X \theta, \lambda_1^-1 D^2 + \lambda_2^-1 I)

        """
        # This needs to be instantiated before building the rest of the Theano
        # graph since self._llik refers to it
        logger.debug('Computing SVD')
        U, D, Vh = numpy.linalg.svd(X)
        self.D_ = _S(D.astype(_real))

        self.log_lambda1_mean = _S(_Z(1), name='log_lambda1_mean')
        log_lambda1_log_prec = _S(_Z(1), name='log_lambda1_log_prec')
        self.log_lambda1_prec = 1e-3 + T.nnet.softplus(log_lambda1_log_prec)

        self.log_lambda2_mean = _S(_Z(1), name='log_lambda2_mean')
        log_lambda2_log_prec = _S(_Z(1), name='log_lambda2_log_prec')
        self.log_lambda2_prec = 1e-3 + T.nnet.softplus(log_lambda2_log_prec)

        super().__init__(U.T.dot(X), U.T.dot(y), a,
                         hyperparam_means=[self.log_lambda1_mean, self.log_lambda2_mean],
                         hyperparam_log_precs=[log_lambda1_log_prec, log_lambda2_log_prec],
                         **kwargs)

        self.predict = _F(inputs=[self.X], outputs=self.eta_mean, allow_input_downcast=True)
        # Coefficient of determination
        R = 1 - T.sqr(self.y - self.eta_mean).sum() / T.sqr(self.y - self.y.mean()).sum()
        self.score = _F(inputs=[self.X, self.y], outputs=R, allow_input_downcast=True)
        self.loss = _F(inputs=[self.X, self.y], outputs=T.sqr(self.y - self.eta_mean).sum(), allow_input_downcast=True)

    def _llik(self, y, eta, phi_raw):
        phi1 = T.addbroadcast(self.min_prec + T.nnet.softplus(self.log_lambda1_mean + T.sqrt(1 / self.log_lambda1_prec) * phi_raw[:,:1]), 1)
        phi2 = T.addbroadcast(self.min_prec + T.nnet.softplus(self.log_lambda2_mean + T.sqrt(1 / self.log_lambda2_prec) * phi_raw[:,1:]), 1)
        # Careful: we need the precision for each individual, so we have to
        # invert D
        phi = phi1 * T.inv(T.sqr(self.D_)) + phi2
        F = -.5 * (-T.log(phi) + T.sqr(y - eta) * phi)
        return T.mean(T.sum(F, axis=1))
