import numpy
import pytest

import pybslmm.pcgc
import pybslmm.simulation

def _test(p, m, pheno):
    def trial(seed):
        s = pybslmm.simulation.Simulation(p=p, pve=0.5, m=m, seed=seed)
        sampler = getattr(s, 'sample_{}'.format(pheno))
        if pheno == 'gaussian':
            x, y = sampler(n=1000)
            return pybslmm.pcgc.estimate(y, pybslmm.pcgc.grm(x))[0][0]
        elif pheno in ('ascertained_probit', 'case_control'):
            K = 0.01
            x, y = sampler(n=1000, K=K, P=0.5)
            return pybslmm.pcgc.estimate(y, pybslmm.pcgc.grm(x), K=K)[0][0]
        else:
            raise ValueError('Invalid phenotype: {}'.format(pheno))
    pve = [trial(i) for i in range(50)]
    m = numpy.mean(pve)
    se = numpy.std(pve)
    assert m - se <= 0.5 <= m + se

def test_infinitesimal_gaussian():
    _test(p=1000, m=None, pheno='gaussian')

def test_unnormalized_gaussian():
    _test(p=1000, m=1000, pheno='gaussian')

def test_non_infinitesimal_gaussian():
    _test(p=1000, m=100, pheno='gaussian')

def test_infinitesimal_ascertained_probit():
    _test(p=1000, m=None, pheno='ascertained_probit')

def test_unnormalized_ascertained_probit():
    _test(p=1000, m=1000, pheno='ascertained_probit')

def test_non_infinitesimal_ascertained_probit():
    _test(p=1000, m=100, pheno='ascertained_probit')

def test_infinitesimal_case_control():
    _test(p=1000, m=None, pheno='case_control')

def test_unnormalized_case_control():
    _test(p=1000, m=1000, pheno='case_control')

def test_non_infinitesimal_case_control():
    _test(p=1000, m=100, pheno='case_control')
