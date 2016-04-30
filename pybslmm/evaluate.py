"""Evaluate the accuracy of the model

We want to recover the correct pi and tau, and also the correct posterior on
theta and z. We use the distribution of parameter estimates in simulated
training sets for the first, and prediction RMSE on a simulated validation set
for the second.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import os
import pickle
import sys

import numpy
import numpy.random
import scipy.special

from .simulation import simulate_ascertained_probit
from .model import fit

def evaluate(seed):
    numpy.random.seed(seed)
    x, y, theta = simulate_ascertained_probit(n=2000, p=10000, K=.01, P=.5,
                                              pve=.5, m=100, batch_size=10000)
    m = numpy.count_nonzero(theta)
    x_train, x_test = x[::2], x[1::2]
    y_train, y_test = y[::2], y[1::2]
    alpha, beta, gamma, pi, tau = fit(x_train, y_train, outer_steps=5, inner_steps=1000,
                                      inner_params={'a': .01})
    rmse = numpy.std(y_test - scipy.special.expit(x_test.dot(alpha * beta)))
    print(pi, tau, rmse)
