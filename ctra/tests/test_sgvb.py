import os

import numpy
import pytest

import ctra

def _fit_model(model, x, y, s, center_y=False, drop=None, **kwargs):
    x_train, x_test = x[::2], x[1::2]
    y_train, y_test = y[::2], y[1::2]
    to_center = [x_train, x_test]
    if center_y:
        to_center.extend([y_train, y_test])
    for data in to_center:
        data -= data.mean(axis=0)
    m = model(x_train, y_train, s.annot, stoch_samples=10, learning_rate=0.1, minibatch_n=50, rho=0.9, random_state=s.random, **kwargs)
    m.fit(max_epochs=100, xv=x_test, yv=y_test)
    if drop is not None:
        weights = numpy.ones(x.shape[1]).astype('int8')
        weights[drop] = 0
        m.w_.set_value(weights)
    return m

def test_gaussian_sgvb_one_component():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_gaussian(n=1000)
        _fit_model(ctra.model.GaussianSGVB, x, y, s, center_y=True)

def test_gaussian_sgvb_drop():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_gaussian(n=1000)
        m = _fit_model(ctra.model.GaussianSGVB, x, y, s, center_y=True, drop=1)

def test_logistic_sgvb_one_component():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_ascertained_probit(n=1000, K=0.01, P=0.5)
        _fit_model(ctra.model.LogisticSGVB, x, y, s)

