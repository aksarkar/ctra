import numpy
import pytest

import ctra.model
import ctra.simulation

def _estimate(simulation, pheno, **kwargs):
    simulation.sample_effects(pve=0.5, **kwargs)
    if pheno == 'gaussian':
        K = None
        x, y = simulation.sample_gaussian(n=1000)
    elif pheno in ('ascertained_probit', 'case_control'):
        K = 0.01
        x, y = getattr(simulation, 'sample_{}'.format(pheno))(n=1000, K=K, P=0.5)
    else:
        raise ValueError('Invalid phenotype: {}'.format(pheno))
    grm = ctra.model.grm(x, simulation.annot)
    h = ctra.model.pcgc(y, grm, K=K)
    return h.sum()

def _sampling_dist(trial_fn):
    pve = [trial_fn(i) for i in range(50)]
    m = numpy.mean(pve)
    se = numpy.std(pve)
    return m, se

def _test(p, pheno, sample_annotations=False, true_annotation=None, **kwargs):
    def trial(seed):
        s = ctra.simulation.Simulation(p=p, seed=seed)
        if sample_annotations:
            s.sample_annotations(proportion=numpy.repeat(0.5, 2))
        elif true_annotation is not None:
            s.load_annotations(true_annotation)
        return _estimate(s, pheno, **kwargs)
    m, se = _sampling_dist(trial)
    assert m - se <= 0.5 <= m + se

def test_inf_gaussian():
    _test(p=1000, pheno='gaussian')

def test_non_inf_gaussian():
    _test(p=1000, pheno='gaussian', annotation_params=[(100, 1)])

def test_inf_ascertained_probit():
    _test(p=1000, pheno='ascertained_probit')

def test_non_inf_ascertained_probit():
    _test(p=1000, pheno='ascertained_probit', annotation_params=[(100, 1)])

def test_inf_case_control():
    _test(p=1000, pheno='case_control')

def test_non_inf_case_control():
    _test(p=1000, pheno='case_control', annotation_params=[(100, 1)])

def test_inf_gaussian_two_degenerate_components():
    _test(p=1000, pheno='gaussian', sample_annotations=True)

def test_non_inf_gaussian_two_degenerate_components():
    _test(p=1000, pheno='gaussian', sample_annotations=True,
          annotation_params=[(100, 1), (100, 1)])

def test_non_inf_case_control_two_degenerate_components():
    _test(p=1000, pheno='case_control', sample_annotations=True,
          annotation_params=[(100, 1), (100, 1)])

def test_non_inf_gaussian_two_components_uenq_prop():
    _test(p=1000, pheno='gaussian', sample_annotations=True,
          annotation_params=[(200, 1), (100, 1)])

def test_non_inf_case_control_two_degenerate_components_uneq_prop():
    _test(p=1000, pheno='case_control', sample_annotations=True,
          annotation_params=[(200, 1), (100, 1)])

def test_non_inf_gaussian_two_components_uenq_scale():
    _test(p=1000, pheno='gaussian', sample_annotations=True,
          annotation_params=[(100, 2), (100, 1)])

def test_non_inf_case_control_two_degenerate_components_uneq_scale():
    _test(p=1000, pheno='case_control', sample_annotations=True,
          annotation_params=[(100, 2), (100, 1)])

def test_true_annotation_one_component():
    p = 1000
    _test(p=p, pheno='gaussian', true_annotation=numpy.zeros(p))

def test_non_inf_true_annotation_one_component():
    p = 1000
    _test(p=p, pheno='gaussian', true_annotation=numpy.zeros(p),
          annotation_params=[(100, 1)])

def test_true_annotation_two_degenerate_components():
    p = 1000
    a = numpy.zeros(p)
    a[500:] = numpy.ones(500)
    _test(p=p, pheno='gaussian', true_annotation=a)

def test_true_annotation_two_degenerate_components_non_contiguous():
    p = 1000
    a = numpy.zeros(p)
    a[::2] = 1
    _test(p=p, pheno='gaussian', true_annotation=a)

def test_true_annotation_two_degenerate_components_non_contiguous_non_inf():
    p = 1000
    a = numpy.zeros(p)
    a[::2] = 1
    _test(p=p, pheno='gaussian', true_annotation=a,
          annotation_params=[(100, 1), (100, 1)])
