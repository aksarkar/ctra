"""Fit variational approximation to desired model.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections

import numpy
import numpy.random
import theano
import theano.printing
import theano.tensor as T

_real = theano.config.floatX
_F = theano.function
_S = theano.shared
_Z = lambda n: numpy.zeros(n).astype(_real)

def fit(X_, y_, num_epochs=5000, max_precision=1e8, a=1e-3, b1=0.9, b2=0.999, e=1e-8):
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

    # Re-parameterization to make expectation differentiable
    random = T.shared_randomstreams.RandomStreams(seed=0)
    eta_raw = random.normal(size=(10, n))
    eta = mu + T.sqrt(nu) * eta_raw

    # Hyperparameters
    pi = (numpy.array(0.1).astype(_real))
    tau = (numpy.array(1).astype(_real))
    hyperparams = [pi, tau]

    # Objective function
    elbo = (
        # Data likelihood
        T.mean(T.sum(y * eta - T.nnet.softplus(eta), axis=1))
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
        if not t % 100:
            print('.', end='')
    print()

    return (alpha.eval(),
            beta.get_value(),
            gamma.eval())

def test(a=1e-3):
    from .simulation import simulate_ascertained_probit
    from sklearn.linear_model import LogisticRegression

    numpy.random.seed(0)
    x, y, theta = simulate_ascertained_probit(n=1000, p=10000, K=.01, P=.5, pve=.5, batch_size=10000)

    alpha, beta, gamma = fit(x[:900,:], y[:900], a=a)
    theta_hat = alpha * beta
    print('RMSE (VB):', numpy.std(y[900:] - (x[900:].dot(alpha * beta) > 0)), end='\n\n')

    alt = LogisticRegression().fit(x[:900,:], y[:900])
    print('RMSE (l2):', numpy.std(y[900:] - alt.predict(x[900:])))
