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
        self.elbo_vals = []

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

class _mvuniform():
    """Multidimensional uniform distribution

    Convenience wrapper to allow sampling m-dimensional vectors using the same
    API as scipy.stats.multivariate_normal

    """
    def __init__(self, a, b, m):
        self.m = m
        self.uniform = scipy.stats.uniform(loc=a, scale=b - a)

    def rvs(self, size):
        return self.uniform.rvs(size=self.m * size).reshape(-1, self.m)

    def logpdf(self, x):
        return self.uniform.logpdf(x).sum()

def _sqdist(a, b, V):
    """Return (a - b)' V^-1 (a - b) for all pairs (a, b)"""
    _D = scipy.spatial.distance.cdist
    return numpy.square(_D(a, numpy.atleast_2d(b),
                           metric='mahalanobis',
                           VI=numpy.linalg.pinv(V)))

class WSABI:
    def __init__(self, m, hyperprior=None, proposal=None, max_retries=10):
        if hyperprior is None:
            self.hyperprior = scipy.stats.multivariate_normal()
        else:
            self.hyperprior = hyperprior
        if proposal is None:
            self.proposal = scipy.stats.multivariate_normal()
        else:
            self.proposal = proposal
        self.m = m
        self.I = numpy.eye(self.m)
        self.gp = None
        # ARD kernel overfits/explodes numerically in our problem
        self.K = GPy.kern.RBF(input_dim=self.m, ARD=False)
        self.x = []
        self.f = []
        self._mean = None
        self._var = None
        self.max_retries = max_retries

    def __iter__(self):
        return self

    def _neg_uncertainty(self, query):
        """Return -V[l(phi) pi(phi)]"""
        query = numpy.atleast_2d(numpy.squeeze(query))
        schur_comp = self.gp.rbf.K(query, self.x).dot(self.Kinv)
        return -(numpy.square(self.hyperprior.pdf(query)) *
                 (self.gp.rbf.K(query) - schur_comp.dot(self.gp.rbf.K(self.x, query))) *
                 (self.offset + .5 * schur_comp.dot(self.f)))

    def __next__(self):
        """Return the next hyperparameter sample phi

        The active sampling scheme finds the point which contributes maximum
        variance to the current estimate of the model evidence.

        """
        logger.debug('Actively sampling next point')
        opt = object()
        n = 0
        while n < self.max_retries and not getattr(opt, 'success', False):
            x0 = self.proposal.rvs(1)
            logger.debug('Starting minimizer from x0={}'.format(x0))
            opt = scipy.optimize.minimize(self._neg_uncertainty, x0=x0, method='Nelder-Mead')
            n += 1
        if not opt.success:
            import pdb; pdb.set_trace()
            raise ValueError('Failed to find next sample')
        return opt.x

    def transform(self, x, f, exp=True):
        """Square root transform function values

        Enforce non-negativity in the chi-square process (squared-GP)
        representing the original function.

        exp - exponentiate log likelihoods

        """
        if exp:
            self.log_scaling = f.max()
            f = numpy.exp(f - self.log_scaling)
        else:
            self.log_scaling = None
        self.offset = .8 * f.min()
        f = numpy.sqrt(2 * (f - self.offset)).reshape(-1, 1)
        self.x = x
        self.f = f
        return self

    def fit(self):
        """Fit the GP on the transformed data"""
        self.gp = GPy.models.GPRegression(self.x, self.f, self.K)
        self.gp.optimize()
        # Pre-compute things
        A = self.gp.rbf.lengthscale * self.I
        Ainv = self.I / self.gp.rbf.lengthscale
        b = self.hyperprior.mean
        B = self.hyperprior.cov
        Binv = numpy.linalg.pinv(B)
        self.Kinv = numpy.linalg.pinv(self.gp.rbf.K(self.x))
        # The term (K^-1 f)^2 comes outside the integrals for posterior
        # mean/variance
        right = numpy.square(self.Kinv.dot(self.f))
        # \int_X K(x, x_d)^2 N(x; b, B) dx
        w = (numpy.square(self.gp.rbf.variance) /
                  numpy.sqrt(numpy.linalg.det(2 * Ainv.dot(B) + self.I)) *
                  numpy.exp(-.5 * _sqdist(self.x, b, .5 * A + B)))

        self._mean = self.offset + .5 * w.T.dot(right)
        if self.log_scaling is not None:
            self._mean = self.log_scaling + numpy.log(self._mean)

        # \int_X K(x, x_d) N(x; b, B) dx
        z = (self.gp.rbf.variance /
             numpy.sqrt(numpy.linalg.det(B)) *
             numpy.exp(-.5 * _sqdist(self.x, b, A + B)))
        # The inner integral for the variance of the marginal likelihood:
        #
        #    \inv_X K(x, x_d) N(x; b, B) K(x, x') dx
        #
        # results in a Gaussian with covariance sigma. The outer integral
        # results in a Gaussian with covariance _lambda
        sigma = Ainv + numpy.linalg.pinv(Ainv + Binv)
        _lambda = numpy.linalg.pinv(sigma) + numpy.linalg.pinv(Ainv + Binv)
        self._var = (self.gp.rbf.variance * z.T.dot(z) *
                     numpy.sqrt(numpy.linalg.det(2 * Ainv + Binv) *
                                numpy.linalg.det(_lambda)) -
                     w.T.dot(self.Kinv).dot(w))
        return self

    def mean(self):
        """Return the posterior mean \int_X m(x) p(x) dx"""
        if self._mean is None:
            raise ValueError('Must fit the model before estimating its mean')
        return self._mean

    def var(self):
        """Return the posterior variance \int_X Cov(x, x') p(x) p(x') dx dx' """
        if self._var is None:
            raise ValueError('Must fit the model before estimating its variance')
        return self._var

    def __repr__(self):
        return 'WSABI(mean={}, var={})'.format(self.mean(), self.var())

class ActiveSampler(Model):
    """Estimate the posterior p(pi, tau | X, y, A) using WSABI-L (Gunter et al., NIPS 2016).

    To evaluate the intractable integral \iint p(X, y, A | pi, tau) p(pi, tau)
    dpi dtau, we put a Gaussian process prior to represent uncertainty in p(X,
    y, A | pi, tau) at points (pi, tau) not yet evaluated. This means we can
    actively choose the next (pi, tau) to minimize the uncertainty in the
    integral (exploration) and maximize the number of high likelihood samples
    (exploitation).

    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def fit(self, init_samples=10, max_samples=100, propose_tau=False, propose_null=False, atol=0.1, **kwargs):
        """Draw samples from the hyperposterior

        init_samples - initial draws from hyperprior to fit GP
        max_samples - maximum number of samples

        """
        pve = self.model.pve
        if propose_null:
            m = 1
        else:
            m = pve.shape[0]
        hyperparam = numpy.zeros((max_samples, m))
        llik = numpy.zeros(max_samples)
        if propose_tau:
            # The naming here is misleading because we rely on self.hyperprior
            # being Gaussian elsewhere, but in the case of proposing tau we
            # specify a log-uniform prior and use importance reweighting to
            # work with a Gaussian prior
            self.hyperprior = scipy.stats.multivariate_normal(cov=numpy.eye(m))
            # Set minimum tau as tau corresponding to pi = 1e-5
            a = -5
            self.proposal = _mvuniform(a, numpy.log(self.model.var_x.sum()), m)
        else:
            self.hyperprior = scipy.stats.multivariate_normal(mean=-2 * numpy.ones(m), cov=2 * numpy.eye(m))
            self.proposal = self.hyperprior
        hyperparam[:init_samples, :] = self.proposal.rvs(size=init_samples).reshape(-1, m)
        self.evidence_gp = WSABI(m=m, hyperprior=self.hyperprior, proposal=self.proposal)
        # In the original problem we need E[g(phi)]:
        #
        #   \int_R g(phi) p(phi | x, y, a) dphi
        # = \int_R g(phi) p(x, y, a | phi) p(phi) / Z dphi
        #
        # Now that we have Z, we can use the same square-root BQ to perform
        # this integration. We don't integrate over Z because its variance is
        # small enough to justify approximating its posterior as a spike.
        self.phi_gp = WSABI(m=m, hyperprior=self.hyperprior, proposal=self.proposal)
        self.params = []
        self.pi_grid = []
        self.tau_grid = []
        for i in range(max_samples):
            if propose_null:
                hyperparam[i] = numpy.repeat(hyperparam[i][0], pve.shape[0])
            if propose_tau:
                tau = numpy.exp(hyperparam[i]).astype(_real)
                pi = numpy.repeat(pve.sum() / (1 - pve.sum()) / (self.model.var_x / tau).sum(),
                                  pve.shape[0]).astype(_real)
            else:
                pi = _expit(hyperparam[i]).astype(_real)
                tau = numpy.repeat(((1 - pve.sum()) * (pi * self.model.var_x).sum()) /
                                   pve.sum(), pve.shape[0]).astype(_real)
            self.pi_grid.append(pi)
            self.tau_grid.append(tau)
            llik[i], _params = self.model.log_weight(pi=pi, tau=tau, **kwargs)
            self.elbo_vals.append(llik[i])
            if propose_tau:
                # Importance-reweighting trick
                llik[i] += (self.proposal.logpdf(hyperparam[i]) -
                            self.hyperprior.logpdf(hyperparam[i]))
            self.params.append(_params)
            if i + 1 >= init_samples:
                logger.debug('Refitting GP for Z')
                self.evidence_gp = self.evidence_gp.transform(hyperparam[:i], llik[:i]).fit()
                logger.debug('Refitting GP for phi')
                self.phi_gp = self.evidence_gp.transform(hyperparam[:i], llik[:i] - self.evidence_gp.mean())
                logger.debug('Sample {}: Z={}, phi={}'.format(len(llik),
                                                              self.evidence_gp,
                                                              self.phi_gp))
                v = self.phi_gp.var()
                if v <= 0:
                    logger.info('Finished active sampling after {} samples (variance vanished)'.format(i))
                    break
                elif v <= atol:
                    logger.info('Finished active sampling after {} samples (tolerance reached)'.format(i))
                    break
                elif i + 1 < max_samples:
                    # Get the next hyperparameter sample
                    hyperparam[i + 1] = next(self.phi_gp)
        # Finalize the fitted hyparameters
        if propose_tau:
            self.tau = self.phi_gp.mean()
            self.pi = numpy.repeat(pve.sum() / (1 - pve.sum()) / (self.model.var_x / self.tau).sum(),
                                   pve.shape[0]).astype(_real)
        else:
            self.pi = self.phi_gp.mean()
            self.tau = numpy.repeat(((1 - pve.sum()) * (self.pi * self.model.var_x).sum()) /
                                    pve.sum(), pve.shape[0]).astype(_real)
        self.pip = []
        return self

    def evidence(self):
        return self.evidence_gp.mean()
