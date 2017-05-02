import os

import numpy
import pytest

import ctra

def _fit_model(model, x, y, s, center_y=False, drop=False):
    x_train, x_test = x[::2], x[1::2]
    y_train, y_test = y[::2], y[1::2]
    to_center = [x_train, x_test]
    if center_y:
        to_center.extend([y_train, y_test])
    for data in to_center:
        data -= data.mean(axis=0)
    m = model(x_train, y_train, s.annot, stoch_samples=10, learning_rate=0.1, minibatch_n=50, rho=0.9, random_state=s.random)
    m.fit(max_epochs=100, xv=x_test, yv=y_test)
    if jacknife:
        m.weights[drop] = 0
    return m

def test_gaussian_sgvb_one_component():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_gaussian(n=1000)
        _fit_model(ctra.model.GaussianSGVB, x, y, s, center_y=True)

def test_gaussian_sgvb_one_component_jacknife():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_gaussian(n=1000)
        m = _fit_model(ctra.model.GaussianSGVB, x, y, s, center_y=True, drop=True)

def test_logistic_sgvb_one_component():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_ascertained_probit(n=1000, K=0.01, P=0.5)
        _fit_model(ctra.model.LogisticSGVB, x, y, s)

