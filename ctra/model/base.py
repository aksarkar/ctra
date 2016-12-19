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

import GPy
import numpy
import scipy.special
import theano

logger = logging.getLogger(__name__)
_real = theano.config.floatX

# We need to change base since we're going to be interpreting the value of pi,
# logit(pi)
_expit = lambda x: scipy.special.expit(x * numpy.log(10))
_logit = lambda x: scipy.special.logit(x) / numpy.log(10)

# This is needed for models implemented as standalone functions rather than
# instance methods
result = collections.namedtuple('result', ['pi', 'pi_grid', 'weights', 'params'])

class Model:
    def __init__(self, X, y, a, pve, **kwargs):
        self.X = X
        self.y = y
        self.a = a
        self.p = numpy.array(list(collections.Counter(self.a).values()), dtype='int')
        self.pve = pve
        self.var_x = numpy.array([X[:,a == i].var(axis=0).sum() for i in range(1 + max(a))])
        self.eps = 1e-8
        self.elbo_vals = None
        logger.info('Fixing parameters {}'.format({'pve': self.pve}))

    def _log_weight(self, params=None, true_causal=None, **hyperparams):
        raise NotImplementedError

    def fit(self, proposals=None, **kwargs):
        raise NotImplementedError
    
    def bayes_factor(self, other):
        """Return the Bayes factor between this model and another fitted model

        Scale importance weights using both sets of un-normalized importance
        samples to account for different baseline ELBO.

        """
        return numpy.exp(self.evidence() - other.evidence())

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
    def __init__(self, X, y, a, pve, **kwargs):
        super().__init__(X, y, a, pve)

    def fit(self, proposals=None, propose_tau=False, **kwargs):
        """Return the posterior mean estimates of the hyperparameters

        proposals - list of proposals for pi
        kwargs - algorithm-specific parameters

        """
        if proposals is None:
            if propose_tau:
                # Propose log(tau)
                proposals = list(itertools.product(*[list(10 ** numpy.arange(-3, 1, 0.5))
                                                    for p_k in self.p]))
            else:
                # Propose logit(pi)
                proposals = list(itertools.product(*[[-10] + list(numpy.arange(_logit(1 / p_k), _logit(.1), .25))
                                                     for p_k in self.p]))
        pve = self.pve

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
                pi[i] = numpy.repeat(pve.sum() / (1 - pve.sum()) / (self.var_x / tau[i]).sum(),
                                     pve.shape[0]).astype(_real)
                assert 0 < pi[i] < 1
            else:
                pi[i] = _expit(numpy.array(prop)).astype(_real)
                tau[i] = numpy.repeat(((1 - pve.sum()) * (pi[i] * self.var_x).sum()) /
                                      pve.sum(), pve.shape[0]).astype(_real)
            elbo_, params_ = self._log_weight(pi=pi[i], tau=tau[i], **kwargs)
            if elbo_ > best_elbo:
                params = params_

        # Perform importance sampling, using ELBO instead of the marginal
        # likelihood
        log_weights = numpy.zeros(shape=len(proposals))
        self.params = []
        logger.info('Performing importance sampling')
        for i, logit_pi in enumerate(proposals):
            log_weights[i], params_ = self._log_weight(pi=pi[i], tau=tau[i], params=params, **kwargs)
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
    def __init__(self, X, y, a, pve):
        super().__init__(X, y, a, pve)
        self.evidence = None

    def fit(self, init_samples=10, max_samples=100, **kwargs):
        m = self.pve.shape
        logit_pi = numpy.zeros(size=(max_samples, m))
        llik = numpy.zeros(max_samples)
        _N = scipy.stats.multivariate_normal
        hyperprior = _N(mean=numpy.zeros(m), covar=numpy.eye(m))
        logit_pi[init_samples] = hyperprior.rvs(size=init_samples)
        K = GPy.kern.RBF(input_dim=m, ARD=True)
        for i in max_samples:
            pi = _expit(logit_pi)
            tau = numpy.repeat(((1 - self.pve.sum()) * (pi * self.var_x).sum()) /
                               self.pve.sum(), self.pve.shape[0]).astype(_real)
            llik[i] = self._log_weight(pi=pi, tau=tau, **kwargs)
            if i > init_samples:
                # Square-root transform the samples. tau is deterministic given
                # pi, so we write do inference on f(phi) = P(X, y, A | phi),
                # g(phi) = sqrt(2 * (f(phi) - alpha))
                phi = logit_pi[:i]
                f_phi = numpy.exp(llik[:i] - llik[:i].max())
                offset = .8 * f_phi.min()
                g_phi = numpy.sqrt(2 * (f_phi - offset))
                # Refit the GP
                self.wsabi = GPy.models.GPRegression(phi, g_phi, K)
                self.wsabi.optimize()
                # Actively sample the next point
                raise NotImplementedError
        # Expected value of the integral. Closed form from Rasmussen &
        # Gharamani, 2003 (eq. 9)
        _K = self.wsabi.rbf.K(phi)
        z = _N(cov=(_K + hyperprior.cov)).pdf(g_phi)
        self.evidence = offset + .5 * sum(wsabi.rbf.variance * z * numpy.linalg.pinv(_K).dot(g_phi))

    def evidence(self):
        if self.evidence is None:
            raise ValueError('Must fit the model before computing Bayes factors')
        return self.evidence
