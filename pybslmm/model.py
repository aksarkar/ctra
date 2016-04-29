"""Fit the hierarchical model

We fit a generalized linear model regressing phenotype against genotype. We
impose a spike-and-slab prior on the coefficients to regularize the
problem. Our inference task is to find the maximum a posteriori estimate of the
parameters pi (probability each SNP is causal) and tau (precision of causal
effects).

pi*, tau* := argmax_{pi, tau} p(x, y | pi, tau)

The inference requires integrating over latent variables z (causal indicator)
and theta (effect size). Our strategy is to fit a variational approximation to
the posterior p(theta, z | x, y) and perform stochastic optimization to find
the MAP estimates. We find the variational approximation by performing an inner
stochastic optimization loop, maximizing the evidence lower bound given the
current values of (pi, tau).

In our stochastic optimization, we use the sample mean of of the individual
sample likelihoods across the random samples eta as a control variate, since
its expectation is 0.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import pdb

import numpy
import numpy.random
import theano
import theano.tensor as T

_real = theano.config.floatX
_F = theano.function
_S = theano.shared
_Z = lambda n: numpy.zeros(n).astype(_real)

def _adam(objective, params, grad=None, a=1e-3, b1=0.9, b2=0.999, e=1e-8):
    """Return a Theano function which updates params according to the gradient

    Adaptive estimation (Kingma & Welling arxiv:1412.6980) tunes the learning
    rate based on exponentially weighted moving averages of the first and
    second moments of the gradient.

    """
    if grad is None:
        grad = T.grad(objective, params)
    epoch = T.iscalar('epoch')
    M = [_S(numpy.zeros_like(param.get_value())) for param in params]
    V = [_S(numpy.zeros_like(param.get_value())) for param in params]
    a_t = a * T.sqrt(1 - T.pow(b2, epoch)) / (1 - T.pow(b1, epoch))
    adam_updates = collections.OrderedDict()
    for p, g, m, v in zip(params, grad, M, V):
        new_m = b1 * m + (1 - b1) * g
        new_v = b2 * v + (1 - b2) * T.sqr(g)
        adam_updates[p] = T.cast(p + a_t * new_m / (T.sqrt(new_v) + e), _real)
        adam_updates[m] = new_m
        adam_updates[v] = new_v
    return _F([epoch], objective, updates=adam_updates)

def logit(y, eta):
    """Return E_q[ln p(y | eta)] assuming a logit link."""
    F = y * eta - T.nnet.softplus(eta)
    return T.mean(T.sum(F, axis=1)) - T.mean(F)

def beta(y, eta, eps=1e-7):
    """Return E_q[ln p(y | eta)] assuming y ~ Beta(sigmoid(eta), exp(-eta))

    The Beta distribution is technically not appropriate for binary responses,
    but we simply clamp the observed values into its support. We might prefer
    Beta response to a binomial response (logit link) because it will handle
    responses close to the decision boundary (E[y] > 0.5) better.

    """
    y = T.clip(y, eps, 1 - eps)
    m = T.clip(T.nnet.sigmoid(eta), eps, 1 - eps)
    v = T.clip(T.exp(-eta), eps, 1 - eps)
    F = (m * v * (T.log(y) - T.log(1 - y)) + v * T.log(1 - y) +
         T.gammaln(v) - T.gammaln(m * v) - T.gammaln((1 - m) * v)
    return T.mean(T.sum(F), axis=1)) - T.mean(F)

    """Return the variational parameters alpha, beta, gamma.

    X_ - dosage matrix (n x p)
    y_ - response vector (n x 1)
    llik - data likelihood under the variational approximation. A function
           which takes parameters y, mu, nu and returns a TensorVariable
    outer_steps - number of hyperparameter stochastic gradient ascent steps
    inner_steps - number of parameter stochastic gradient ascent steps
    max_precision - maximum value of gamma
    a, b1, b2, e - Adam parameters for auto-tuning learning rate

    """
    if X_.shape[0] != y_.shape[0]:
        raise ValueError("dimension mismatch: X {} vs y {}".format(X_.shape, y_.shape))
    n, p = X_.shape

    # Observed data
    X = theano.shared(X_.astype(_real))
    y_obs = theano.shared(y_)
    y = T.cast(y_obs, 'int32')

    # Variational parameters
    alpha_raw = _S(_Z(p))
    alpha = T.nnet.sigmoid(alpha_raw)
    beta = _S(_Z(p))
    gamma_raw = _S(_Z(p))
    gamma = max_precision * T.nnet.sigmoid(gamma_raw)
    params = [alpha_raw, beta, gamma_raw]

    # Variational approximation (re-parameterize eta = X theta)
    mu = T.dot(X, alpha * beta)
    nu = T.dot(T.sqr(X), alpha / gamma + alpha * (1 - alpha) + T.sqr(beta))

    # Re-parameterize to get a differentiable expectation, following Kingma &
    # Welling, ICLR 2014 (http://arxiv.org/abs/1312.6114)
    random = T.shared_randomstreams.RandomStreams(seed=0)
    eta_raw = random.normal(size=(10, y.shape[0]))
    eta = mu + T.sqrt(nu) * eta_raw
    # Take independent samples for the outer optimization
    eta_outer_raw = random.normal(size=(10, y.shape[0]))
    eta_outer = mu + T.sqrt(nu) * eta_outer_raw

    # Hyperparameters
    pi_raw = _S(numpy.array(numpy.log(.1 / .9)).astype(_real))
    pi = T.nnet.sigmoid(pi_raw)
    tau_raw = _S(numpy.array(0, dtype=_real))
    tau = T.exp(-tau_raw)
    hyper_params = [pi_raw, tau_raw]

    # Outer objective function
    objective = (
        # Data likelihood
        llik(y, eta_outer)
        # Hyperprior
        / pi * T.sqr(tau)
    )

    # Inner objective function
    elbo = (
        # Data log likelihood
        llik(y, eta) +
        # Prior
        + .5 * T.sum(alpha * (1 + T.log(tau) - T.log(gamma) - tau * (T.sqr(beta) + 1 / gamma)))
        # Variational entropy
        - T.sum(alpha * T.log(alpha / pi) + (1 - alpha) * T.log((1 - alpha) / (1 - pi)))
    )

    inner_step = _adam(elbo, params, **adam_params)
    outer_step = _adam(objective, hyper_params, **adam_params)

    # Optimize
    for s in range(outer_steps):
        for t in range(inner_steps):
            elbo = inner_step(t + 1)
        objective = outer_step(s + 1)
        print('Objective:', objective)
    pdb.set_trace()

    return (alpha.eval(),
            beta.get_value(),
            gamma.eval(),
            pi.get_value(),
            tau.get_value())

if __name__ == '__main__':
    import os
    import pickle
    import sys

    # Hack needed for Broad UGER
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LIBRARY_PATH')
    with open(sys.argv[1], 'rb') as f:
        x, y, theta = pickle.load(f)
    m = numpy.count_nonzero(theta)
    x_train, x_test = x[::2], x[1::2]
    y_train, y_test = y[::2], y[1::2]
    fit(x_train, y_train, outer_steps=50, inner_steps=1000)
