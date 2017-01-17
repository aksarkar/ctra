"""Outer level algorithms to estimate the hyperposterior p(pi, tau | X, y, A)

We fit a generalized linear model regressing phenotype against genotype. We
impose a spike-and-slab prior on the coefficients to regularize the
problem. Our inference task is to estimate the posterior distribution of the
parameter pi (probability each SNP is causal).

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import itertools
import logging

import numpy
import scipy.optimize
import scipy.spatial
import scipy.special
import theano
import GPy

logger = logging.getLogger(__name__)
_real = theano.config.floatX

# Remove GPy noise
logging.getLogger("GP").setLevel('ERROR')
logging.getLogger("GPRegression").setLevel('ERROR')

# We need to change base since we're going to be interpreting the value of pi,
# logit(pi)
_expit = lambda x: scipy.special.expit(x * numpy.log(10))
_logit = lambda x: scipy.special.logit(x) / numpy.log(10)

# This is needed for models implemented as standalone functions rather than
# instance methods
result = collections.namedtuple('result', ['pi', 'pi_grid', 'weights', 'params'])

class Model:
    def __init__(self, model, **kwargs):
        self.model = model
        self.elbo_vals = None

    def fit(self, **kwargs):
        raise NotImplementedError
    
    def evidence(self):
        raise NotImplementedError

    def bayes_factor(self, other):
        """Return the Bayes factor between this model and another fitted model

        Scale importance weights using both sets of un-normalized importance
        samples to account for different baseline ELBO.

        """
        return numpy.ravel(numpy.exp(self.evidence() - other.evidence()))[0]

class Algorithm:
    def __init__(self, X, y, a, pve, **kwargs):
        self.X = X
        self.y = y
        self.a = a
        self.p = numpy.array(list(collections.Counter(self.a).values()), dtype='int')
        self.pve = pve
        self.var_x = numpy.array([X[:,a == i].var(axis=0).sum() for i in range(1 + max(a))])
        self.eps = 1e-8
        logger.info('Fixing parameters {}'.format({'pve': self.pve}))

    def log_weight(self, pi, tau, **kwargs):
        raise NotImplementedError

class ImportanceSampler(Model):
    """Estimate the posterior p(pi, tau | X, y, A) using un-normalized importance
sampling.

    This is the strategy taken by Carbonetto & Stephens, Bayesian Anal 2012 to
    avoid computing the intractable normalizer, assuming uniform prior
    p(pi)p(tau) and uniform proposal distribution q(pi)q(tau) (in practice,
    taking samples on a fixed grid). Bound the intractable model evidence term
    in the importance weight is by finding the optimal variational
    approximation to the intractable posterior p(theta, z | x, y, a).

    """
    def __init__(self, model, **kwargs):
        super().__init__(model)

    def fit(self, proposals=None, propose_tau=False, **kwargs):
        """Return the posterior mean estimates of the hyperparameters

        proposals - list of proposals for pi
        kwargs - algorithm-specific parameters

        """
        if proposals is None:
            if propose_tau:
                # Propose log(tau)
                proposals = list(itertools.product(*[list(10 ** numpy.arange(-3, 1, 0.5))
                                                    for p_k in self.model.p]))
            else:
                # Propose logit(pi)
                proposals = list(itertools.product(*[[-10] + list(numpy.arange(_logit(1 / p_k), _logit(.1), .25))
                                                     for p_k in self.model.p]))
        pve = self.model.pve

        if 'true_causal' in kwargs:
            z = kwargs['true_causal']
            kwargs['true_causal'] = numpy.clip(z.astype(_real), 1e-4, 1 - 1e-4)
            assert not numpy.isclose(max(kwargs['true_causal']), 1)

        # Find the best initialization of the variational parameters. In
        # general we need to warm start different sets of parameters per model,
        # so in this generic implementation we can't unpack things.
        params = None
        pi = numpy.zeros(shape=(len(proposals), pve.shape[0]), dtype=_real)
        tau = numpy.zeros(shape=pi.shape, dtype=_real)
        best_elbo = float('-inf')
        logger.info('Finding best initialization')
        for i, prop in enumerate(proposals):
            # Proposing one of (pi, tau) induces a proposal for the other (Guan
            # et al., Ann Appl Stat 2011; Carbonetto et al., Bayesian Anal
            # 2012)
            if propose_tau:
                tau[i] = numpy.array(prop).astype(_real)
                pi[i] = numpy.repeat(pve.sum() / (1 - pve.sum()) / (self.model.var_x / tau[i]).sum(),
                                     pve.shape[0]).astype(_real)
                assert 0 < pi[i] < 1
            else:
                pi[i] = _expit(numpy.array(prop)).astype(_real)
                tau[i] = numpy.repeat(((1 - pve.sum()) * (pi[i] * self.model.var_x).sum()) /
                                      pve.sum(), pve.shape[0]).astype(_real)
            elbo_, params_ = self.model.log_weight(pi=pi[i], tau=tau[i], **kwargs)
            if elbo_ > best_elbo:
                params = params_

        # Perform importance sampling, using ELBO instead of the marginal
        # likelihood
        log_weights = numpy.zeros(shape=len(proposals))
        self.params = []
        logger.info('Performing importance sampling')
        for i, logit_pi in enumerate(proposals):
            log_weights[i], params_ = self.model.log_weight(pi=pi[i], tau=tau[i], params=params, **kwargs)
            self.params.append(params_)

        self.elbo_vals = log_weights.copy()
        # Scale the log importance weights before normalizing to avoid numerical
        # problems
        log_weights -= max(log_weights)
        normalized_weights = numpy.exp(log_weights - scipy.misc.logsumexp(log_weights))
        self.weights = normalized_weights
        self.pip = normalized_weights.dot(numpy.array([alpha for alpha, *_ in self.params]))
        self.pi_grid = pi
        self.tau_grid = tau
        self.pi = normalized_weights.dot(pi)
        self.tau = normalized_weights.dot(tau)
        return self

    def evidence(self):
        if self.elbo_vals is None:
            raise ValueError('Must fit the model before computing Bayes factors')
        return (numpy.log(numpy.exp(self.elbo_vals - self.elbo_vals.max()).mean()) +
                self.elbo_vals.max())

class BayesianQuadrature(Model):
    """Estimate the posterior p(pi, tau | X, y, A) using Bayesian quadrature.

    The idea is described in Gunter et al., NIPS 2016. To evaluate the
    intractable integral \iint p(X, y, A | pi, tau) p(pi, tau) dpi dtau, we put
    a Gaussian process prior to represent uncertainty in p(X, y, A | pi, tau)
    at points (pi, tau) not yet evaluated. This means we can actively choose
    the next (pi, tau) to minimize the uncertainty in the integral
    (exploration) and maximize the number of high likelihood samples
    (exploitation).

    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._evidence = None

    def _neg_exp_var_evidence(self, query, phi, g_phi):
        """Return -V[l(phi) pi(phi)]"""
        query = numpy.atleast_2d(numpy.squeeze(query))
        schur_comp = self.wsabi.rbf.K(query, phi).dot(self._Kinv)
        return -(numpy.square(self.hyperprior.pdf(query)) *
                 (self.wsabi.rbf.K(query) - schur_comp.dot(self.wsabi.rbf.K(phi, query))) *
                 (self._offset + .5 * schur_comp.dot(g_phi)))

    def _active_sample(self, phi, g_phi, max_retries=20):
        """Return the next hyperparameter sample phi

        The active sampling scheme finds the point which contributes maximum
        variance to the current estimate of the model evidence.

        """
        logger.debug('Actively sampling next point')
        opt = object()
        n = 0
        while n < max_retries and not getattr(opt, 'success', False):
            x0 = self.hyperprior.rvs(1)
            logger.debug('Starting minimizer from x0={}'.format(x0))
            opt = scipy.optimize.minimize(self._neg_exp_var_evidence, x0=x0, args=(phi, g_phi))
        if not opt.success:
            raise ValueError('Failed to find next sample')
        return opt.x

    def _update_params(self, llik, phi, g_phi, propose_tau=False):
        """Update the relevant derived parameters after seeing design points phi

        The parameters are expectations and variances over the quadrature GP.

        """
        pve = self.model.pve
        m = pve.shape[0]
        A = numpy.diag(self.wsabi.rbf.lengthscale)
        Ainv = numpy.diag(1 / self.wsabi.rbf.lengthscale)
        b = self.hyperprior.mean
        B = self.hyperprior.cov
        Binv = numpy.linalg.pinv(B)
        I = numpy.eye(Ainv.shape[0])
        # \int_X K(x, x_d) N(x; b, B) dx
        z = (self.wsabi.rbf.variance /
             numpy.sqrt(numpy.linalg.det(B)) *
             numpy.exp(-.5 * numpy.square(scipy.spatial.distance.cdist(phi, numpy.atleast_2d(b), metric='mahalanobis', VI=numpy.linalg.pinv(A + B)))))
        # \int_X K(x, x_d)^2 N(x; b, B) dx
        w = (numpy.square(self.wsabi.rbf.variance) /
             numpy.sqrt(numpy.linalg.det(2 * Ainv.dot(B) + I)) *
             numpy.exp(-.5 * numpy.square(scipy.spatial.distance.cdist(phi, numpy.atleast_2d(b), metric='mahalanobis', VI=numpy.linalg.pinv(.5 * A + B)))))
        self._Kinv = numpy.linalg.pinv(self.wsabi.rbf.K(phi))
        # The inner integral for the variance of the marginal likelihood:
        #
        #    \inv_X K(x, x_d) N(x; b, B) K(x, x') dx
        #
        # results in a Gaussian with covariance sigma. The outer integral
        # results in a Gaussian with covariance _lambda
        sigma = Ainv + numpy.linalg.pinv(Ainv + Binv)
        _lambda = numpy.linalg.pinv(sigma) + numpy.linalg.pinv(Ainv + Binv)
        # The term (K^-1 f)^2 comes outside the integrals
        v = numpy.square(self._Kinv.dot(g_phi))
        self._evidence = (llik.max() + numpy.log(self._offset + .5 * w.T.dot(v)))
        self.evidence_var = (self.wsabi.rbf.variance * z.T.dot(z) * numpy.sqrt(numpy.linalg.det(2 * Ainv + Binv) * numpy.linalg.det(_lambda)) -
                             w.T.dot(self._Kinv).dot(w))
        self.elbo_vals = llik
        if propose_tau:
            self.tau = numpy.exp(numpy.exp(llik - llik.max() - scipy.misc.logsumexp(llik - llik.max())).dot(phi))
            self.pi = numpy.repeat(pve.sum() / (1 - pve.sum()) / (self.model.var_x / self.tau).sum(),
                                   pve.shape[0]).astype(_real)
        else:
            self.pi = _expit(numpy.exp(llik - llik.max() - scipy.misc.logsumexp(llik - llik.max())).dot(phi))
            self.tau = numpy.repeat(((1 - pve.sum()) * (self.pi * self.model.var_x).sum()) /
                                    pve.sum(), pve.shape[0]).astype(_real)
        if not numpy.isfinite(self.evidence_var):
            import pdb; pdb.set_trace()
        logger.debug('Sample {}: evidence = {}, variance = {}, pi = {}'.format(len(llik), self.evidence, self.evidence_var, self.pi))

    def fit(self, init_samples=10, max_samples=100, propose_tau=False, **kwargs):
        """Draw samples from the hyperposterior

        init_samples - initial draws from hyperprior to fit GP
        max_samples - maximum number of samples

        """
        pve = self.model.pve
        m = pve.shape[0]
        hyperparam = numpy.zeros((max_samples, m))
        llik = numpy.zeros(max_samples)
        self.hyperprior = scipy.stats.multivariate_normal(mean=numpy.zeros(m), cov=4 * numpy.eye(m))
        hyperparam[:init_samples,:] = self.hyperprior.rvs(size=init_samples).reshape(-1, m)
        K = GPy.kern.RBF(input_dim=m, ARD=True)
        self.params = []
        self.pi_grid = []
        self.tau_grid = []
        for i in range(max_samples):
            if propose_tau:
                tau = numpy.exp(hyperparam[i])
                pi = numpy.repeat(pve.sum() / (1 - pve.sum()) / (self.model.var_x / tau).sum(),
                                  pve.shape[0]).astype(_real)
            else:
                pi = _expit(hyperparam[i])
                tau = numpy.repeat(((1 - pve.sum()) * (pi * self.model.var_x).sum()) /
                                   pve.sum(), pve.shape[0]).astype(_real)
            self.pi_grid.append(pi)
            self.tau_grid.append(tau)
            llik[i], _params = self.model.log_weight(pi=pi, tau=tau, **kwargs)
            self.params.append(_params)
            if i + 1 >= init_samples:
                logger.debug('Refitting the quadrature GP')
                # Square-root transform the (hyperparameter, llik) pairs. tau
                # is deterministic given pi, so we do inference on f(phi) =
                # P(X, y, A | phi), g(phi) = sqrt(2 * (f(phi) - alpha))
                phi = hyperparam[:i]
                f_phi = numpy.exp(llik[:i] - llik[:i].max())
                self._offset = .8 * f_phi.min()
                g_phi = numpy.sqrt(2 * (f_phi - self._offset)).reshape(-1, 1)
                # Refit the GP
                self.wsabi = GPy.models.GPRegression(phi, g_phi, K)
                self.wsabi.optimize()
                self._update_params(llik[:i], phi, g_phi, propose_tau)
                if self.evidence_var <= 0:
                    logger.info('Finished active sampling after {} samples (variance vanished)'.format(i))
                    self.evidence_var = 0
                    break
                elif i + 1 < max_samples:
                    # Get the next hyperparameter sample
                    hyperparam[i + 1] = self._active_sample(phi, g_phi)
        return self

    def evidence(self):
        if self._evidence is None:
            raise ValueError('Must fit the model before computing Bayes factors')
        return self._evidence
