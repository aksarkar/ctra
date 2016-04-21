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

class Model:
    def __init__(self, n, p):
        self.shape = (n, p)
        self.random = T.shared_randomstreams.RandomStreams(seed=0)

        X = T.matrix('X')
        y = T.lvector('y')
        self.obs = [X, y]

        # Variational parameters
        alpha_raw = _S(_Z(p))
        alpha = T.nnet.sigmoid(alpha_raw)
        beta = _S(_Z(p))

        # Genuine prior on variance of causal effects
        self.gamma_max = 1.5
        gamma_raw = _S(_Z(p))
        gamma = self.gamma_max * T.nnet.sigmoid(gamma_raw)
        self.params = [alpha_raw, beta, gamma_raw]

        # Variational approximation (re-parameterize eta = X theta)
        mu = T.dot(X, alpha * beta)
        nu = T.dot(T.sqr(X), alpha / gamma + alpha * (1 - alpha) + T.sqr(beta))

        # Re-parameterization to make expectation differentiable
        eta_raw = self.random.normal(size=(10, n))
        eta = mu + T.sqrt(nu) * eta_raw

        # Hyperparameters
        pi = _S(numpy.array(0.1).astype(_real))
        tau = _S(numpy.array(1).astype(_real))
        self.hyperparams = [pi, tau]

        # Objective function
        self.elbo = (
            # Data likelihood
            T.mean(T.sum(y * eta - T.nnet.softplus(eta), axis=1))
            # Prior
            + .5 * T.sum(alpha * (1 + T.log(tau) - T.log(gamma) - tau * alpha * (T.sqr(beta) + 1 / gamma)))
            # Variational entropy
            - T.sum(alpha * T.log(alpha / pi) + (1 - alpha) * T.log((1 - alpha) / (1 - pi)))
        )

    def fit(self, X_, y_, num_epochs=1000, a=1e-3, b1=0.9, b2=0.999, e=1e-3):
        if X_.shape != self.shape:
            raise ValueError("dimension mismatch: model {} vs X {}".format(self.shape, X_.shape))
        if X_.shape[0] != y_.shape[0]:
            raise ValueError("dimension mismatch: X {} vs y {}".format(X_.shape, y_.shape))

        # Gradient ascent (Adam)
        grad = T.grad(self.elbo, self.params)
        epoch = T.iscalar('epoch')
        M = [_S(_Z(param.get_value().shape))
             for param in self.params]
        V = [_S(_Z(param.get_value().shape))
             for param in self.params]
        a_t = a * T.sqrt(1 - T.pow(b2, epoch)) / (1 - T.pow(b1, epoch))
        grad_updates = collections.OrderedDict()
        for param, g, m, v in zip(self.params, grad, M, V):
            new_m = b1 * m + (1 - b1) * g
            new_v = b2 * v + (1 - b2) * T.sqr(g)
            grad_updates[param] = param + a_t * new_m / (T.sqrt(new_v) + e)
            grad_updates[m] = new_m
            grad_updates[v] = new_v

        # Observed data
        X = theano.shared(X_.astype(_real))
        y = theano.shared(y_)

        givens = list(zip(self.obs, [X, y]))
        self.adam_step = _F([epoch], self.elbo, updates=grad_updates, givens=givens)

        # Optimize
        for epoch in range(1, num_epochs + 1):
            elbo = self.adam_step(epoch)
            # elbo = self.eb_step()
            if not epoch % 100:
                print('ELBO:', elbo)

        self.alpha = T.nnet.sigmoid(self.params[0].get_value()).eval()
        self.beta = self.params[1].get_value()
        self.gamma = self.gamma_max * T.nnet.sigmoid(self.params[2].get_value()).eval()
        self.pi, self.tau = [p.get_value() for p in self.hyperparams]
        self.elbo_val = elbo
        return self

def test():
    from .simulation import simulate
    numpy.random.seed(0)
    x, y, theta = simulate(1000, 100, .5)
    m = Model(*x.shape).fit(x, y)
    theta_hat = m.alpha * m.beta
    return m, x, y, theta
