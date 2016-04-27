"""Fit variational approximation to desired model.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections

import numpy
import numpy.random
import theano
import theano.tensor as T

_real = theano.config.floatX
_F = theano.function
_S = theano.shared
_Z = lambda n: numpy.zeros(n).astype(_real)

def logit(y, eta):
    """Return E_q[ln p(y | eta)] assuming a logit link."""
    return T.mean(T.sum(y * eta - T.nnet.softplus(eta), axis=1))

def beta(y, eta, eps=1e-7):
    """Return E_q[ln p(y | eta)] assuming y ~ Beta(sigmoid(eta), exp(-eta))

    The Beta distribution is technically not appropriate for binary responses,
    but we simply clamp the observed values into its support. We might prefer
    Beta response to a binomial response (logit link) because it will handle
    responses close to the decision boundary (E[y] > 0.5) better.

    """
    y = T.clip(y, eps, 1 - eps)
    m = T.nnet.sigmoid(eta)
    v = T.exp(-eta)
    return T.mean(m * v * (T.log(y) - T.log(1 - y)) + v * T.log(1 - y) +
                  T.gammaln(v) - T.gammaln(m * v) - T.gammaln((1 - m) * v))

def fit(X_, y_, llik=logit, num_epochs=5000, max_precision=1e6, a=1e-3, b1=0.9, b2=0.999, e=1e-8):
    """Return the variational parameters alpha, beta, gamma.

    X_ - dosage matrix (n x p)
    y_ - response vector (n x 1)
    llik - data likelihood under the variational approximation. A function
           which takes parameters y, mu, nu and returns a TensorVariable
    num_epochs - number of stochastic gradient steps
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

    # Hyperparameters
    pi = (numpy.array(0.1).astype(_real))
    tau = (numpy.array(1).astype(_real))
    hyperparams = [pi, tau]

    # Objective function
    elbo = (
        # Data likelihood
        llik(y, eta) +
        # Prior
        + .5 * T.sum(alpha * (1 + T.log(tau) - T.log(gamma) - tau * (T.sqr(beta) + 1 / gamma)))
        # Variational entropy
        - T.sum(alpha * T.log(alpha / pi) + (1 - alpha) * T.log((1 - alpha) / (1 - pi)))
    )

    # Gradient ascent (Adam)
    grad = T.grad(elbo, params)
    epoch = T.iscalar('epoch')
    M = [_S(_Z(p)) for param in params]
    V = [_S(_Z(p)) for param in params]
    a_t = a * T.sqrt(1 - T.pow(b2, epoch)) / (1 - T.pow(b1, epoch))
    adam_updates = collections.OrderedDict()
    for p, g, m, v in zip(params, grad, M, V):
        new_m = b1 * m + (1 - b1) * g
        new_v = b2 * v + (1 - b2) * T.sqr(g)
        adam_updates[p] = T.cast(p + a_t * new_m / (T.sqrt(new_v) + e), _real)
        adam_updates[m] = new_m
        adam_updates[v] = new_v
    adam_step = _F([epoch], elbo, updates=adam_updates)

    # Optimize
    for t in range(num_epochs):
        elbo = adam_step(t + 1)

    return (alpha.eval(),
            beta.get_value(),
            gamma.eval())

if __name__ == '__main__':
    import os
    import pickle
    import pdb
    import sys

    from .simulation import simulate_ascertained_probit
    from sklearn.linear_model import LogisticRegression

    # Hack needed for Broad UGER
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LIBRARY_PATH')
    with open(sys.argv[1], 'rb') as f:
        x, y, theta = pickle.load(f)

    m = numpy.count_nonzero(theta)
    x_train, x_test = x[::2], x[1::2]
    y_train, y_test = y[::2], y[1::2]

    alpha, beta, gamma = fit(x_train, y_train)
    alt = LogisticRegression(fit_intercept=False).fit(x_train, y_train)

    pdb.set_trace()
