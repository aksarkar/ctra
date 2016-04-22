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

def fit(X_, y_, minibatch_size, num_epochs=5000, a=1e-3, b1=0.9, b2=0.999, e=1e-3):
    if X_.shape[0] != y_.shape[0]:
        raise ValueError("dimension mismatch: X {} vs y {}".format(X_.shape, y_.shape))
    n, p = X_.shape

    # Observed data
    X = theano.shared(X_.astype(_real))
    num_minibatch = p // minibatch_size
    y = theano.shared(y_)

    # Variational parameters
    alpha_raw = _S(_Z(p))  # alpha = sigmoid(alpha_raw)
    beta = _S(_Z(p))
    gamma_raw = _S(_Z(p))  # gamma = gamma_max * sigmoid(gamma_raw)
    params = [alpha_raw, beta, gamma_raw]

    # Minibatches of SNPs (features)
    minibatch_index = T.iscalar()
    minibatch_start = minibatch_index * minibatch_size
    minibatch_end = (minibatch_index + 1) * minibatch_size
    X_s = X[:,minibatch_start:minibatch_end]
    minibatch_params = [p[minibatch_start:minibatch_end] for p in params]
    alpha_raw_s, beta_s, gamma_raw_s = minibatch_params

    # Re-parameterize to guarantee bounded values
    alpha_s = T.nnet.sigmoid(alpha_raw_s)
    gamma_max = 1.5
    gamma_s = gamma_max * T.nnet.sigmoid(gamma_raw_s)

    # Variational approximation (re-parameterize eta = X theta)
    mu = T.dot(X_s, alpha_s * beta_s)
    nu = T.dot(T.sqr(X_s), alpha_s / gamma_s + alpha_s * (1 - alpha_s) + T.sqr(beta_s))

    # Re-parameterization to make expectation differentiable
    random = T.shared_randomstreams.RandomStreams(seed=0)
    eta_raw = random.normal(size=(10, n))
    eta = mu + T.sqrt(nu) * eta_raw

    # Hyperparameters
    pi = _S(numpy.array(0.1).astype(_real))
    # Genuine prior on P(theta | z)
    tau = _S(numpy.array(1.5).astype(_real))
    hyperparams = [pi, tau]

    # Objective function
    elbo = (
        # Data likelihood
        T.mean(T.sum(y * eta - T.nnet.softplus(eta), axis=1))
        # Prior
        + .5 * T.sum(alpha_s * (1 + T.log(tau) - T.log(gamma_s) - tau * alpha_s * (T.sqr(beta_s) + 1 / gamma_s)))
        # Variational entropy
        - T.sum(alpha_s * T.log(alpha_s / pi) + (1 - alpha_s) * T.log((1 - alpha_s) / (1 - pi)))
    )

    # Gradient ascent (Adam)
    grad = T.grad(elbo, minibatch_params)
    epoch = T.iscalar('epoch')
    M = [_S(_Z(minibatch_size)) for param in minibatch_params]
    V = [_S(_Z(minibatch_size)) for param in minibatch_params]
    a_t = a * T.sqrt(1 - T.pow(b2, epoch)) / (1 - T.pow(b1, epoch))
    adam_updates = collections.OrderedDict()
    for p, p_s, g, m, v in zip(params, minibatch_params, grad, M, V):
        new_m = b1 * m + (1 - b1) * g
        new_v = b2 * v + (1 - b2) * T.sqr(g)
        adam_updates[p] = T.inc_subtensor(p_s, a_t * new_m / (T.sqrt(new_v) + e))
        adam_updates[m] = new_m
        adam_updates[v] = new_v
    adam_step = _F([minibatch_index, epoch], elbo, updates=adam_updates)

    # Optimize
    for t in range(num_epochs):
        elbo = adam_step(t % num_minibatch, t + 1)
        if not t % 100:
            print('ELBO:', elbo)

    return (T.nnet.sigmoid(alpha_raw.get_value()).eval(),
            beta.get_value(),
            gamma_max * T.nnet.sigmoid(gamma_raw.get_value()).eval())

def test():
    from .simulation import simulate
    import sklearn.linear_model

    numpy.random.seed(0)
    x, y, theta = simulate(1000, 10000, .5)
    alpha, beta, gamma = fit(x[:900,:], y[:900], 10000)
    yhat = x[900:].dot(alpha * beta)
    rmse = numpy.std(y[900:] - yhat)
    print('RMSE:', rmse)

    ytilde = sklearn.linear_model.LogisticRegression().fit(x[:900,:], y[:900]).predict(x[900:])
    print('RMSE (l2):', numpy.std(y[900:] - ytilde))
