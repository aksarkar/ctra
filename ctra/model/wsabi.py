"""Warped sequential approximate Bayesian inference (Gunter et al., NIPS 2016)

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import logging

import numpy
import scipy.optimize
import scipy.stats
import scipy.spatial
import GPy

logger = logging.getLogger(__name__)

def _sqdist(a, b, V):
    """Return (a - b)' V^-1 (a - b) for all pairs (a, b)"""
    _D = scipy.spatial.distance.cdist
    return numpy.square(_D(a, numpy.atleast_2d(b),
                           metric='mahalanobis',
                           VI=numpy.linalg.pinv(V)))

class GP:
    """Base class for wrappers around GPy classes"""
    def __init__(self, m, hyperprior):
        self.m = m
        self.hyperprior = hyperprior
        self.I = numpy.eye(self.m)
        self.x = []
        self.f = []
        self.gp = None
        self._mean = None
        self._var = None
        self.Kinv = None

    def fit(self, x, y, log_scaling, offset):
        """Fit the GP to the data and estimate the required integrals"""
        raise NotImplementedError

    def schur_complement(self, query):
        """Return K(query, f)' K^{-1}"""
        raise NotImplementedError

    def mean(self):
        """Return the posterior mean \int_X m(x) p(x) dx"""
        if self._mean is None:
            raise ValueError('Must fit the model before estimating its mean')
        return numpy.squeeze(self._mean)

    def var(self):
        """Return the posterior variance \int_X Cov(x, x') p(x) p(x') dx dx' """
        if self._var is None:
            raise ValueError('Must fit the model before estimating its variance')
        return numpy.squeeze(self._var)

    def neg_uncertainty(self, query):
        """Return -V[l(phi) pi(phi)]"""
        if self.gp is None:
            raise ValueError('Must fit the model before estimating uncertainty')
        query = numpy.atleast_2d(numpy.squeeze(query))
        K_x_query = self.gp.kern.K(query, self.x)
        schur_comp = K_x_query.dot(self.Kinv)
        return -(numpy.square(self.hyperprior.pdf(query)) *
                 (self.gp.kern.K(query) - schur_comp.dot(K_x_query.T)) *
                 (self.offset + .5 * schur_comp.dot(self.f)))

    def __repr__(self):
        return '{}(mean={}, var={})'.format(self.__name__, self.mean(), self.var())

class GPRBF(GP):
    """Wrapper around GPy.models.GPRegression(x, y, GPy.kern.RBF)"""
    def __init__(self, m, hyperprior, max_lengthscale=1):
        super().__init__(m, hyperprior)
        # ARD kernel overfits/explodes numerically in our problem
        self.K = GPy.kern.RBF(input_dim=self.m, ARD=False)
        # Get typical lengthscale from simulation
        self.K.lengthscale.constrain_bounded(0, max_lengthscale)

    def fit(self, x, y):
        """Fit the GP on the transformed data"""
        self.x = x
        self.f = y
        self.gp = GPy.models.GPRegression(self.x, self.f, self.K)
        self.gp.optimize_restarts()
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
        self._mean = self.log_scaling + numpy.log(self.offset + w.T.dot(right))
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

class WSABI:
    """Warped sequential approximate Bayesian inference (Gunter et al., NIPS 2016)

    To evaluate the intractable integral \int p(x | theta) p(theta) dtheta, we
    put a Gaussian process prior to represent uncertainty in p(x | theta) at
    points theta not yet evaluated. This means we can actively choose the next
    theta to minimize the uncertainty in the integral (exploration) and
    maximize the number of high likelihood samples (exploitation).

    """
    def __init__(self, m, hyperprior=None, proposal=None, max_retries=10, gp=None):
        if hyperprior is None:
            self.hyperprior = scipy.stats.multivariate_normal()
        else:
            self.hyperprior = hyperprior
        if proposal is None:
            self.proposal = scipy.stats.multivariate_normal()
        else:
            self.proposal = proposal
        if gp is None:
            self.gp = GPRBF(m, self.hyperprior)
        else:
            self.gp = gp
        self.max_retries = max_retries

    def __iter__(self):
        return self

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
            opt = scipy.optimize.minimize(self.gp.neg_uncertainty, x0=x0, method='Nelder-Mead')
            n += 1
        if not opt.success:
            raise ValueError('Failed to find next sample')
        return opt.x

    def fit(self, x, f):
        """Square root transform function values and fit the GP

        Enforce non-negativity in the chi-square process (squared-GP)
        representing the original function.

        exp - exponentiate log likelihoods

        """
        self.gp.log_scaling = f.max()
        f = numpy.exp(f[:] - self.gp.log_scaling)
        self.gp.offset = .8 * f.min()
        f = numpy.sqrt(2 * (f - self.gp.offset)).reshape(-1, 1)
        x = numpy.atleast_2d(x)
        self.gp.fit(x, f)
        return self

    def mean(self):
        """Return the posterior mean \int_X m(x) p(x) dx"""
        return self.gp.mean()

    def var(self):
        """Return the posterior variance \int_X Cov(x, x') p(x) p(x') dx dx' """
        return self.gp.var()

    def __repr__(self):
        return 'WSABI(mean={}, var={})'.format(self.mean(), self.var())
