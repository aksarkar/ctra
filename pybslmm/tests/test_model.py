import os

import pytest

import pybslmm.model
import pybslmm.simulation

@pytest.fixture
def fix_environment():
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LIBRARY_PATH')

def test_one_task(fix_environment):
    n=2000
    p=10000
    K=.01
    P=.5
    pve=0.5
    seed=0
    annotation_params = [(100, 1), (50, 1)]
    with pybslmm.simulation.simulation(p, pve, annotation_params, seed) as s:
        x, y = s.sample_case_control(n=n, K=K, P=P)
        x_train, x_test = x[::2], x[1::2]
        y_train, y_test = y[::2], y[1::2]
        a = s.annot
        model = pybslmm.model.Model(x, y, a)
        alpha, beta, pi, tau = model.sgvb()
