"""Importance sampling to estimate the hyperposterior

We fit a generalized linear model regressing phenotype against genotype. We
impose a spike-and-slab prior on the coefficients to regularize the
problem. Our inference task is to estimate the posterior distribution of the
parameter pi (probability each SNP is causal).

We use un-normalized importance sampling to estimate the posterior, avoiding
the intractable normalizer. We replace the model evidence term in the
importance weight with the optimal lower bound achieved by a variational
approximation to the intractable posterior p(theta, z | x, y, a).

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import itertools
import logging

import numpy
import scipy.special
import theano

logger = logging.getLogger(__name__)
_real = theano.config.floatX

# We need to change base since we're going to be interpreting the value of pi,
# logit(pi)
_expit = lambda x: scipy.special.expit(x * numpy.log(10))
_logit = lambda x: scipy.special.logit(x) / numpy.log(10)

class ImportanceSampler():
    def __init__(self, X, y, a, pve, **kwargs):
        self.X = X
        self.y = y
        self.a = a
        self.pve = pve
        self.var_x = numpy.array([X[:,a == i].var(axis=0).sum() for i in range(1 + max(a))])
        self.eps = numpy.finfo(float).eps
        logger.info('Fixing parameters {}'.format({'pve': self.pve}))

    def _log_weight(self, params=None, **hyperparams):
        raise NotImplementedError

    def fit(self, **kwargs):
        """Return the posterior mean estimates of the hyperparameters

        kwargs - algorithm-specific parameters

        """
        # Propose pi_0, pi_1. This induces a proposal for tau_0, tau_1
        # following Guan et al. Ann Appl Stat 2011; Carbonetto et al. Bayesian
        # Anal 2012
        proposals = list(itertools.product(*[numpy.arange(-3, 0.5, 0.5)
                                             for a in self.pve]))

        # Find the best initialization of the variational parameters. In
        # general we need to warm start different sets of parameters per model,
        # so in this generic implementation we can't unpack things.
        params = None
        pi = numpy.zeros(shape=(len(proposals), self.pve.shape[0]), dtype=_real)
        tau = numpy.zeros(shape=pi.shape, dtype=_real)
        logit_pi_prior = scipy.stats.norm(loc=-3).logpdf
        best_elbo = float('-inf')
        logger.info('Finding best initialization')
        for i, logit_pi in enumerate(proposals):
            pi[i] = _expit(numpy.array(logit_pi)).astype(_real)
            tau[i] = (((1 - self.pve) * pi[i] * self.var_x) / self.pve).astype(_real)
            elbo_, params_ = self._log_weight(pi=pi[i], tau=tau[i], **kwargs)
            elbo_ += logit_pi_prior(logit_pi).sum()
            if elbo_ > best_elbo:
                params = params_

        # Perform importance sampling, using ELBO instead of the marginal
        # likelihood
        log_weights = numpy.zeros(shape=len(proposals))
        logger.info('Performing importance sampling')
        for i, logit_pi in enumerate(proposals):
            log_weights[i], *_ = self._log_weight(pi=pi[i], tau=tau[i],
                                                  params=params, **kwargs)
            log_weights[i] += logit_pi_prior(logit_pi).sum()

        # Scale the log importance weights before normalizing to avoid numerical
        # problems
        log_weights -= max(log_weights)
        normalized_weights = numpy.exp(log_weights - scipy.misc.logsumexp(log_weights))
        self.weights = normalized_weights
        self.pi_grid = pi
        self.pi = normalized_weights.dot(pi)
        return self
