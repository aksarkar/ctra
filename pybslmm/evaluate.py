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
from sklearn.linear_model import LogisticRegression

from .simulation import *
from .model import fit

def prc(y, p):
    for thresh in sorted(p):
        y_hat = (p > thresh).astype('int')
        yield sum(y * y_hat) / sum(y + y_hat), sum(y * y_hat) / sum(y)

def auprc(y, p):
    points = numpy.array(sorted(prc(y, p)))
    delta = numpy.diff(points[:,0], axis=0)
    return points[1:,1].dot(delta)

def evaluate(datafile=None, seed=0, pve=0.5, m=100):
    if datafile is not None:
        with open(datafile, 'rb') as f:
            x, y, a, theta = pickle.load(f)
    else:
        numpy.random.seed(seed)
        x, y, theta = sample_ascertained_probit(n=2000, p=10000, K=.01, P=.5,
                                                pve=pve, m=m, batch_size=10000)
    x_train, x_test = x[::2], x[1::2]
    y_train, y_test = y[::2], y[1::2]
    baseline = auprc(y_test, LogisticRegression(fit_intercept=False).fit(x_train, y_train).predict_proba(x_test)[:,1])
    result = None
    pi = 0.1 + numpy.zeros(2)
    tau = 1 + numpy.zeros(2)
    for step in fit(x_train, y_train, a, pi, tau):
        result = step
    alpha, beta, gamma = result
    comparison = auprc(y_test, scipy.special.expit(x_test.dot(alpha * beta)))
    print(baseline, comparison)

if __name__ == '__main__':
    evaluate(datafile=sys.argv[1])
