"""Coordinate ascent to optimize the varational objective

Generalize the coordinate ascent algorithm of Carbonetto and Stephens, Bayesian
Anal (2012).

Just de-referencing the hyperparameters works even for the multiple annotation
case because the local updates just refer to the shared hyperparameter.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import logging

import numpy
import scipy.special

from .base import Algorithm

logger = logging.getLogger(__name__)
_R = numpy.random
_logit = lambda x: scipy.special.logit(x) / numpy.log(10)

class GaussianCoordinateAscent(Algorithm):
    def __init__(self, X, y, a, pve, **kwargs):
        super().__init__(X, y, a, pve)

    def log_weight(self, pi, tau, params=None, atol=1e-4, true_causal=None, **hyperparams):
        X = self.X
        y = self.y
        a = self.a
        n, p = X.shape
        y = y.reshape(-1, 1)
        pi_deref = numpy.choose(a, pi)
        logit_pi = _logit(pi_deref)
        tau_deref = numpy.choose(a, tau)
        # Initial configuration
        if params is None:
            if true_causal is not None:
                alpha = true_causal
            else:
                alpha = _R.uniform(size=p)
                alpha /= alpha.sum()
            beta = _R.normal(size=p)
            sigma2 = y.var()
        else:
            alpha_, beta_, gamma, sigma2_ = params
            alpha = alpha_.copy()
            beta = beta_.copy()
            sigma2 = sigma2_.copy()
        logger.debug('Starting coordinate ascent given {}'.format({'pve': self.pve,
                                                                   'pi': pi,
                                                                   'tau': tau,
                                                                   'sigma2': sigma2,
                                                                   'var_x': self.var_x}))
        # Precompute things
        eps = numpy.finfo(float).eps
        xty = X.T.dot(y)
        xtx_jj = numpy.einsum('ij,ji->i', X.T, X)
        gamma = (xtx_jj + tau_deref) / sigma2
        eta = X.dot(alpha * beta).reshape(-1, 1)
        # Coordinate ascent
        L = numpy.log
        elbo = float('-inf')
        converged = False
        reverse = False
        logger.debug('{:>8s} {:>8s} {:>8s}'.format('ELBO', 'sigma2', 'Vars'))
        while not converged:
            alpha_ = alpha.copy()
            beta_ = beta.copy()
            elbo_ = elbo
            for j in range(p):
                xj = X[:, j].reshape(-1, 1)
                theta_j = alpha[j] * beta[j]
                beta[j] = (xty[j] + xtx_jj[j] * theta_j - (xj * eta).sum()) / (gamma[j] * sigma2)
                ssr = beta[j] * beta[j] * gamma[j]
                if true_causal is None:
                    alpha[j] = numpy.clip(scipy.special.expit(numpy.log(10) * logit_pi[j] + .5 * (ssr + L(tau_deref[j]) - L(sigma2) - L(gamma[j]))), eps, 1 - eps)
                eta += xj * (alpha[j] * beta[j] - theta_j)
            sigma2 = (numpy.square(y - eta).sum() +
                      (xtx_jj * alpha * (1 / gamma + (1 - alpha) * beta ** 2)).sum() +
                      (alpha.dot(tau_deref * (1 / gamma + numpy.square(beta))))) / (n + alpha.sum())
            gamma = (xtx_jj + tau_deref) / sigma2
            elbo = (-.5 * (n * L(2 * numpy.pi * sigma2) +
                            numpy.square(y - eta).sum() / sigma2 +
                            (xtx_jj * alpha * (1 / gamma + (1 - alpha) * beta ** 2)).sum() / sigma2) +
                     .5 * (alpha * (1 + L(tau_deref) - L(gamma) - L(sigma2) -
                                    tau_deref / sigma2 * (numpy.square(beta) + 1 / gamma))).sum() -
                     (alpha * L(alpha / pi_deref) +
                      (1 - alpha) * L((1 - alpha) / (1 - pi_deref))).sum())
            logger.debug('{:>8.6g} {:>8.3g} {:>8.0f}'.format(elbo, sigma2, alpha.sum()))
            if not numpy.isfinite(elbo):
                raise ValueError('ELBO must be finite')
            elif elbo > 0:
                raise ValueError('ELBO must be non-positive')
            if elbo < elbo_:
                converged = True
            elif (numpy.isclose(alpha_, alpha, atol=atol).all() and
                  numpy.isclose(alpha_ * beta_, alpha * beta, atol=atol).all()):
                converged = True
        return elbo, (alpha, beta, gamma, sigma2)

class LogisticCoordinateAscent(Algorithm):
    def __init__(self, X, y, a, pve, **kwargs):
        super().__init__(X, y, a, pve)
    
    def _update_logistic_var_params(self, zeta):
        # Variational lower bound to the logistic likelihood (Jaakola & Jordan,
        # 2000)
        d = (scipy.special.expit(zeta) - .5) / (zeta + self.eps)
        bias = (self.y - .5).sum() / d.sum()
        yhat = self.y - .5 - bias * d
        xty = numpy.einsum('ij,i->j', self.X, yhat)
        xd = numpy.einsum('ij,i->j', self.X, d)
        xdx = numpy.einsum('ij,i,ij->j', self.X, d, self.X) - numpy.square(xd) / d.sum()
        return d, yhat, xty, xd, xdx

    def log_weight(self, pi, tau, params=None, atol=1e-4, true_causal=None, **hyperparams):
        X = self.X
        y = self.y
        a = self.a
        n, p = X.shape
        y = y.reshape(-1, 1)
        pi_deref = numpy.choose(a, pi)
        logit_pi = _logit(pi_deref)
        tau_deref = numpy.choose(a, tau)
        # Initial configuration
        if params is None:
            if true_causal is not None:
                alpha = true_causal
            else:
                alpha = _R.uniform(size=p)
                alpha /= alpha.sum()
            beta = _R.normal(size=p)
            zeta = numpy.ones(n)
        else:
            alpha_, beta_, _, zeta_ = params
            alpha = alpha_.copy()
            beta = beta_.copy()
            zeta = zeta_.copy()
        logger.debug('Starting coordinate ascent given {}'.format({'pve': self.pve, 'pi': pi, 'tau': tau}))
        # Precompute things
        eta = X.dot(alpha * beta).reshape(-1, 1)
        d, yhat, xty, xd, xdx = self._update_logistic_var_params(zeta)
        gamma = xdx + tau_deref
        # Coordinate ascent
        L = numpy.log
        S = numpy.square
        elbo = float('-inf')
        converged = False
        reverse = False
        logger.debug('{:>8s} {:>8s}'.format('ELBO', 'Vars'))
        while not converged:
            alpha_ = alpha.copy()
            beta_ = beta.copy()
            zeta_ = zeta.copy()
            elbo_ = elbo
            for j in range(p):
                xj = X[:, j].reshape(-1, 1)
                theta_j = alpha[j] * beta[j]
                beta[j] = (xty[j] + xdx[j] * theta_j + xd[j] * d.T.dot(eta) / d.sum() - xj.T.dot(numpy.diag(d)).dot(eta)) / gamma[j]
                ssr = beta[j] * beta[j] * gamma[j]
                if true_causal is None:
                    alpha[j] = numpy.clip(scipy.special.expit(numpy.log(10) * logit_pi[j] + .5 * (ssr + L(tau_deref[j]) - L(gamma[j]))), self.eps, 1 - self.eps)
                eta += xj * (alpha[j] * beta[j] - theta_j)
            gamma = xdx + tau_deref
            a = 1 / d.sum()
            # M step for free parameters zeta
            bias = a * ((y - 0.5).sum() - d.dot(eta))
            # V[theta_j] under the variational approximation
            theta_var = alpha * (1 / gamma + (1 - alpha) * beta ** 2)
            bias_var = a * (1 + a * (theta_var * S(xd)).sum())
            theta_bias_covar = -a * xd * theta_var
            zeta = numpy.sqrt(S(bias + eta).ravel() + bias_var +
                              numpy.einsum('ij,j,ij->i', X, theta_var, X) +
                              2 * X.dot(theta_bias_covar))
            d, yhat, xty, xd, xdx = self._update_logistic_var_params(zeta)
            elbo = (L(a) / 2 + a * S((y - 0.5).sum()) / 2 + L(scipy.special.expit(zeta)).sum() +
                    numpy.square(d.dot(eta)) / (2 * d.sum()) +
                    zeta.dot(d * zeta - 1) / 2 + yhat.T.dot(eta) - numpy.einsum('ji,jk,ki', eta, numpy.diag(d), eta) +
                    -.5 * xdx.dot(theta_var) +
                     .5 * (alpha * (1 + L(tau_deref) - L(gamma) - tau_deref * (S(beta) + 1 / gamma))).sum() -
                     (alpha * L(alpha / pi_deref) + (1 - alpha) * L((1 - alpha) / (1 - pi_deref))).sum())
            logger.debug('{:>8.6g} {:>8.0f}'.format(elbo[0], alpha.sum()))
            if not numpy.isfinite(elbo):
                import pdb; pdb.set_trace()
                raise ValueError('ELBO must be finite')
            elif elbo > 0:
                raise ValueError('ELBO must be non-positive')
            if elbo < elbo_:
                converged = True
            elif (numpy.isclose(alpha_, alpha, atol=atol).all() and
                  numpy.isclose(alpha_ * beta_, alpha * beta, atol=atol).all()):
                converged = True
        return elbo, (alpha, beta, gamma, zeta)
