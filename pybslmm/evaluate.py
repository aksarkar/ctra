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

from .simulation import sample_case_control
from .model import fit

def evaluate(datafile=None, seed=0, pve=0.5, m=100):
    if datafile is not None:
        with open(datafile, 'rb') as f:
            x, y, theta = pickle.load(f)
    else:
        numpy.random.seed(seed)
        x, y, theta = sample_case_control(n=2000, p=10000, K=.01, P=.5,
                                          pve=pve, m=m, batch_size=10000)
    m = numpy.count_nonzero(theta)
    x_train, x_test = x[::2], x[1::2]
    y_train, y_test = y[::2], y[1::2]
    for alpha, beta, gamma, pi, tau in fit(x_train, y_train, steps=10000):
        rmse = numpy.std(y_test - scipy.special.expit(x_test.dot(alpha * beta)))
        print(pi, tau, rmse)

if __name__ == '__main__':
    evaluate(datafile=sys.argv[1])
