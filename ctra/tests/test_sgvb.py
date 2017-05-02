import os

import pytest

import ctra

def _fit_model(model, x, y, center_y=False):
        x_train, x_test = x[::2], x[1::2]
        y_train, y_test = y[::2], y[1::2]
        to_center = [x_train, x_test]
        if center_y:
            to_center.extend([y_train, y_test])
        for data in to_center:
            data -= data.mean(axis=0)
        m = model(x_train, y_train, s.annot, stoch_samples=10, learning_rate=0.1, minibatch_size=50, rho=0.9, random_state=s.random)
        m.fit(max_epochs=100, xv=x_test, yv=y_test)

def test_gaussian_one_component():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_gaussian(n=500)
        _fit_model(ctra.model.GaussianSGVB, x, y, center_y=True)

def test_logistic_one_component():
    with ctra.simulation.simulation(p=2000, pve=0.5, annotation_params=[(200, 1)], seed=0) as s:
        x, y = s.sample_ascertained_probit(n=500)
        _fit_model(ctra.model.LogisticSGVB, x, y)
