"""Evaluate the accuracy of the model

We want to recover the correct pi and tau, and also the correct posterior on
theta and z. We use the distribution of parameter estimates in simulated
training sets for the first, and AUPRC on a simulated validation set for the
second.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import pickle
import sys

import numpy
import scipy.special
import sklearn.linear_model

import ctra.pcgc
import ctra.model
import ctra.simulation

def prc(y, p):
    """Precision-recall curve"""
    for thresh in sorted(p):
        y_hat = (p > thresh).astype('int')
        yield sum(y * y_hat) / sum(y + y_hat), sum(y * y_hat) / sum(y)

def auprc(y, p):
    """Estimate area under the precision-recall curve using a Riemann sum"""
    points = numpy.array(sorted(prc(y, p)))
    delta = numpy.diff(points[:,0], axis=0)
    return points[1:,1].dot(delta)

def evaluate_gaussian_vb(n=2000, p=10000, pve=0.5, seed=0):
    annotation_params = [(100, 1), (50, 1)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_gaussian(n=n)
        x_train, x_test = x[::2], x[1::2]
        y_train, y_test = y[::2], y[1::2]
        a = s.annot
        elbo, alpha, beta = ctra.model.fit_gaussian(x_train, y_train, a,
                                                    pi=numpy.array([.1, .05]),
                                                    tau=numpy.array([1, 1]),
                                                    sigma2=0.5, alpha=None,
                                                    beta=None)
        baseline = numpy.std(y_test - sklearn.linear_model.ElasticNet().fit(x_train, y_train).predict(x_test))
        comparison = numpy.std(y_test - x_test.dot(alpha * beta))
        print(baseline, comparison)

def evaluate_gaussian_is(n=2000, p=10000, pve=0.5, seed=0):
    annotation_params = [(100, 1), (50, 1)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_gaussian(n=n)
        x_train, x_test = x[::2], x[1::2]
        y_train, y_test = y[::2], y[1::2]
        a = s.annot
        alpha, beta, pi, tau = ctra.model.importance_sampling(x_train, y_train, a)
        rmse_null = numpy.std(y_test - numpy.mean(y_train))
        rmse_opt = numpy.std(y_test - x_test.dot(s.theta))
        rmse_enet = numpy.std(y_test - sklearn.linear_model.ElasticNet().fit(x_train, y_train).predict(x_test))
        rmse_is = numpy.std(y_test - x_test.dot(alpha * beta))
        print('RMSE (elasticnet) =', rmse_enet)
        print('RPG (elasticnet) =', (rmse_null - rmse_enet) / (rmse_null - rmse_opt))
        print('RMSE (IS) =', rmse_is)
        print('RPG (IS) =', (rmse_null - rmse_is) / (rmse_null - rmse_opt))
        print('Posterior mean pi={}, tau={}'.format(pi, tau))

def evaluate_sgvb(n=2000, p=10000, K=.01, P=.5, pve=0.5, seed=0):
    annotation_params = [(100, 1), (50, 1)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_case_control(n=n, K=K, P=P)
        x_train, x_test = x[::2], x[1::2]
        y_train, y_test = y[::2], y[1::2]
        a = s.annot
        alpha, beta, pi, tau = ctra.model.fit(x_train, y_train, a)
        baseline = auprc(y_test, x_test.dot(ctra.pcgc.logit_regression(x_train, y_train).x))
        comparison = auprc(y_test, scipy.special.expit(x_test.dot(alpha * beta)))
        print(baseline, comparison)

def evaluate_pcgc_two_components(n=1000, p=1000, pve=0.5):
    """Use PCGC to compute "heritability enrichment" under different architectures

    Two equal-sized components, with:
    1. Twice as many causal variants in first component
    2. Double variance of causal effects in first component

    """
    def trial(seed, annotation_params, n=n, p=p, pve=pve):
        """Return enrichment of PVE in two component model"""
        s = ctra.simulation.Simulation(p=p, seed=seed)
        s.sample_annotations(proportion=numpy.repeat(0.5, 2))
        s.sample_effects(pve=pve, annotation_params=annotation_params)
        x, y = s.sample_gaussian(n=n)
        pve = ctra.pcgc.estimate(y, ctra.pcgc.grm(x, s.annot))
        return pve

    def dist(num_trials, annotation_params):
        estimates = numpy.array([trial(seed, annotation_params)
                                 for seed in range(num_trials)])
        pve = estimates.mean(axis=0)
        se = estimates.std(axis=0)
        enrichment = pve / (pve.sum() * 0.5)
        return pve, se, enrichment

    print('double_num_causal', dist(50, annotation_params=[(200, 1), (100, 1)]))
    print('double_causal_effect', dist(50, annotation_params=[(100, 2), (100, 1)]))
