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

def evaluate_gaussian_is(n=2000, p=10000, pve=0.5, seed=0):
    annotation_params = [(100, 1), (50, 1)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_gaussian(n=n)
        m = ctra.model.GaussianModel(x, y, s.annot).fit()
        print(annotation_params, m.pve, m.pi)

def evaluate_gaussian_equal_prop(n=2000, p=1000, pve=0.25, seed=0):
    annotation_params = [(100, 1), (100, 0.5)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_gaussian(n=n)
        m = ctra.model.GaussianModel(x, y, s.annot).fit()
        print(annotation_params, m.pve, m.pi)

def evaluate_gaussian_binary(n=2000, p=1000, pve=0.25, K=0.01, P=0.5, seed=0):
    annotation_params = [(100, 1), (50, 1)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_case_control(n=n, K=K, P=P)
        m = ctra.model.GaussianModel(x, y, s.annot, K=K).fit(atol=1e-2)
        print(annotation_params, m.pve, m.pi)

def evaluate_gaussian_binary_equal_prop(n=2000, p=1000, pve=0.25, K=0.01, P=0.5, seed=0):
    annotation_params = [(100, 1), (100, 0.5)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_case_control(n=n, K=K, P=P)
        m = ctra.model.GaussianModel(x, y, s.annot, K=K).fit(atol=1e-2)
        print(annotation_params, m.pve, m.pi)

def test_known_data():
    with open('genotypes.txt') as f:
        x = numpy.loadtxt(f, delimiter=',')
    with open('phenotype.txt') as f:
        y = numpy.loadtxt(f)
    w = (y > 0).astype('int')
    a = numpy.zeros(x.shape[1], dtype='int8')
    m = ctra.model.GaussianModel(x, y, a).fit()
    print(m.pi)

def evaluate_sgvb(n=2000, p=1000, K=.01, P=.5, pve=0.5, seed=0):
    annotation_params = [(100, 1), (50, 1)]
    with ctra.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_case_control(n=n, K=K, P=P)
        a = numpy.zeros(p, dtype='int8')
        m = ctra.model.LogisticModel(x, y, a, K).fit()
        print('pi={:.3g}'.format(m.pi))

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
