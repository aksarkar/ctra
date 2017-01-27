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

import matplotlib
import numpy
import scipy.special
import theano
import GPy

import ctra.model.wsabi

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

matplotlib.pyplot.switch_backend('pdf')

class Model:
    def __init__(self, model, **kwargs):
        self.model = model
        self.elbo_vals = []
        # In general we need to warm start different sets of parameters per
        # model, so in the generic implementation we can't unpack things.
        self.params = []
        self.pi_grid = []
        self.tau_grid = []

    def propose(self, hyper, propose_tau=False, pool=True, **kwargs):
        """Propose pi (tau), and return the tuple (pi, tau) consistent with
self.model.pve

        Assuming PVE fixed, proposing one of (pi, tau) induces a proposal for
        the other (Guan et al., Ann Appl Stat 2011; Carbonetto et al., Bayesian
        Anal 2012).

        """
        # Proposing one of (pi, tau) induces a proposal for the other (Guan
        # et al., Ann Appl Stat 2011; Carbonetto et al., Bayesian Anal
        # 2012)
        pve = self.model.pve
        if propose_tau:
            tau = numpy.atleast_1d(10 ** numpy.atleast_1d(hyper))
            if pool:
                pi = numpy.repeat(pve.sum() / (1 - pve.sum()) / (self.model.var_x / tau).sum(),
                                  pve.shape[0])
            else:
                pi = pve / (1 - pve) / (self.model.var_x / tau)
        else:
            pi = _expit(numpy.atleast_1d(hyper))
            if pool:
                tau = numpy.repeat(((1 - pve.sum()) * (pi * self.model.var_x).sum()) /
                                   pve.sum(), pve.shape[0]).astype(_real)
            else:
                tau = ((1 - pve) * pi * self.model.var_x) / pve
        self.pi_grid.append(pi.astype(_real))
        self.tau_grid.append(tau.astype(_real))

    def _handle_converged(self):
        # Scale the log importance weights before normalizing to avoid numerical
        # problems
        self.weights = self.elbo_vals - max(self.elbo_vals)
        self.weights = numpy.exp(self.weights - scipy.misc.logsumexp(self.weights))
        self.pip = self.weights.dot(numpy.array([alpha for alpha, *_ in self.params]))
        self.theta = self.weights.dot(numpy.array([alpha * beta for alpha, beta, *_ in self.params]))
        self.pi = self.weights.dot(self.pi_grid)
        self.tau = self.weights.dot(self.tau_grid)

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

    def predict(self, x):
        """Return the posterior mean prediction"""
        return x.dot(self.theta)

    def score(self, x, y):
        """Return the coefficient of determination of the model fit"""
        return 1 - (numpy.square(self.predict(x) - y).sum() /
                    numpy.square(y - y.mean()).sum())

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

    def fit(self, proposals=None, propose_tau=False, pool=True, **kwargs):
        """Return the posterior mean estimates of the hyperparameters

        proposals - list of proposals for pi
        kwargs - algorithm-specific parameters

        """
        if proposals is None:
            if propose_tau:
                # Propose log(tau^-1)
                proposals = list(itertools.product(*[list(numpy.arange(0, 3, 0.5))
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

        if 'params' in kwargs:
            logger.info('Using specified initialization')
            params = kwargs['params']
        else:
            best_elbo = float('-inf')
            logger.info('Finding best initialization')
            for hyper in proposals:
                self.propose(hyper, propose_tau=propose_tau, pool=pool)
                elbo_, params_ = self.model.log_weight(pi=self.pi_grid[-1], tau=self.tau_grid[-1], **kwargs)
                if elbo_ > best_elbo:
                    params = params_
            kwargs['params'] = params

        logger.info('Performing importance sampling')
        for pi, tau in zip(self.pi_grid, self.tau_grid):
            elbo_, params_ = self.model.log_weight(pi=pi, tau=tau, **kwargs)
            self.elbo_vals.append(elbo_)
            self.params.append(params_)

        self._handle_converged()
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

class ActiveSampler(Model):
    """Estimate the posterior p(pi, tau | X, y, A) using WSABI-L (Gunter et al.,
NIPS 2016)."""
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def fit(self, init_samples=10, max_samples=40, propose_tau=False,
            propose_null=False, pool=True, vtol=0.1, **kwargs):
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
            # Set minimum log10(tau) as log10(tau) corresponding to pi = 1e-5. Set
            # maximum log10(tau) as log10(tau) corresponding to pi = 1
            self.proposal = _mvuniform(-5, numpy.log10(self.model.var_x.sum()), m)
        else:
            self.hyperprior = scipy.stats.multivariate_normal(mean=-2 * numpy.ones(m), cov=2 * numpy.eye(m))
            self.proposal = self.hyperprior
        hyperparam[:init_samples, :] = self.proposal.rvs(size=init_samples).reshape(-1, m)
        self.evidence_gp = ctra.model.wsabi.WSABI(m=m, hyperprior=self.hyperprior, proposal=self.proposal)
        for i in range(max_samples):
            if propose_null:
                hyperparam[i] = numpy.repeat(hyperparam[i][0], pve.shape[0])
            self.propose(hyperparam[i], propose_tau=propose_tau, pool=pool)
            llik[i], _params = self.model.log_weight(pi=self.pi_grid[-1], tau=self.tau_grid[-1], **kwargs)
            # This is needed for the IS estimator
            self.elbo_vals.append(llik[i])
            if propose_tau:
                # Importance-reweighting trick
                llik[i] += (self.proposal.logpdf(hyperparam[i]) -
                            self.hyperprior.logpdf(hyperparam[i]))
            self.params.append(_params)
            if i + 1 >= init_samples:
                logger.debug('Refitting GP for Z')
                self.evidence_gp = self.evidence_gp.fit(hyperparam[:i], llik[:i])
                logger.info('Sample {}: phi={}, Z={}'.format(i + 1, hyperparam[i], self.evidence_gp))
                v = self.evidence_gp.var()
                if v <= 0:
                    logger.info('Finished active sampling after {} samples (variance vanished)'.format(i))
                    break
                elif v <= vtol:
                    logger.info('Finished active sampling after {} samples (tolerance reached)'.format(i))
                    break
                elif i + 1 < max_samples:
                    # Get the next hyperparameter sample
                    hyperparam[i + 1] = next(self.evidence_gp)
        self._handle_converged()
        return self

    def evidence(self):
        return self.evidence_gp.mean()
