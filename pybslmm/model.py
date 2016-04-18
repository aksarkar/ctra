import collections

import numpy
import theano
import theano.config
import theano.tensor as T

def _adam(objective, params, a, b1, b2, e):

class Model:
    _S = theano.shared
    _F = theano.function
    _Z = lambda n: numpy.zeros(n).astype(theano.config.floatX)

    def __init__(self, a=1e-3, b1=0.9, b2=0.999, e=1e-3):
        self.random = T.shared_randomstreams.RandomStreams(seed=0)

        X = T.matrix('X')
        y = T.matrix('y')

        # Variational parameters
        n, p = X.shape
        alpha = _S(_Z(p), name='alpha')
        beta = _S(_Z(p), name='beta')
        gamma = _S(_Z(p), name='gamma')
        self.params = [alpha, beta, gamma]

        # Variational approximation (re-parameterize eta = X theta)
        mu = T.dot(X, alpha * beta)
        nu = T.dot(X * X, (alpha / gamma + alpha * (1 - alpha) + beta * beta))
        eta_raw = self.random.normal(n)
        eta = mu + T.pow(nu, .5) * eta_raw

        # Generative model parameters
        pi = T.scalar('pi')
        tau = T.scalar('tau')
        self.hyperparams = [pi, tau]

        # Objective function
        elbo = T.mean(
            # Data likelihood
            T.dot(y, eta) - T.log(1 + T.exp(T.dot(y, eta)))
            # Data prior
            - T.sum(alpha * T.log(alpha / pi) + (1 - alpha) T.log(1 - alpha / (1 - pi)))
            # Variational likelihood
            -.5 * T.sum(T.log(nu) + (eta - mu)^2 / nu)
            # Variational prior
            -.5 * T.sum(alpha * (1 + T.log(tau) - T.log(gamma) -
                                 tau * alpha * (beta * beta + 1 / gamma)))
        )

        # Stochastic gradient parameter updates (Adam)
        grad = T.grad(elbo, self.)
        epoch = T.iscalar('epoch')
        M = [_S(_Z(param.get_value().shape), name='m{}'.format(param.name))
             for param in self.params]
        V = [_S(_Z(param.get_value().shape), name='v{}'.format(param.name))
             for param in self.params]
        a_t = a * T.sqrt(1 - b2 ** epoch) / (1 - b1 ** epoch)
        updates = {}
        for param, g, m, v in zip(self.params, grad, M, V):
            new_m = b1 * m + (1 - b1) * g
            new_v = b2 * v + (1 - b2) * (g ** 2)
            updates[param.name] = param + a_t * new_m / (T.sqrt(new_v) + e)
            updates[m.name] = new_m
            updates[v.name] = new_v
        self.sg_step = theano.function([X, y, epoch], elbo, updates=updates)

        # Hyperparameter updates
        eb_updates = {'pi': T.sum(alpha) / p,
                      'tau': T.sum(alpha) / T.sum(alpha * alpha * (beta * beta + 1 / gamma))}
        self.eb_step = theano.function([], elbo, updates=eb_updates)

    def fit(self, X, y, minibatch_size=1000, num_epochs=100):
        for epoch in range(num_epochs):
            self.sg_step(X, y)
            self.eb_step()
