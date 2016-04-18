import collections

import numpy
import theano
import theano.tensor as T

_real = theano.config.floatX
_S = theano.shared

class Model:
    def __init__(self, n, p):
        self.shape = (n, p)
        self.random = T.shared_randomstreams.RandomStreams(seed=0)

        X = T.matrix('X')
        y = T.vector('y')
        self.obs = [X, y]

        # Variational parameters
        _Z = lambda n: numpy.zeros(n).astype(_real)
        alpha = _S(_Z(p), name='alpha')
        beta = _S(_Z(p), name='beta')
        gamma = _S(_Z(p), name='gamma')
        self.params = [alpha, beta, gamma]

        # Variational approximation (re-parameterize eta = X theta)
        mu = T.dot(X, alpha * beta)
        nu = T.dot(X * X, (alpha / gamma + alpha * (1 - alpha) + beta * beta))
        eta_raw = self.random.normal((n,))
        eta = mu + T.pow(nu, .5) * eta_raw

        # Generative model parameters
        pi = _S(numpy.array(0).astype(_real), name='pi')
        tau = _S(numpy.array(0).astype(_real), name='tau')
        self.hyperparams = [pi, tau]

        # Objective function
        self.elbo = T.mean(
            # Data likelihood
            T.sum(T.log(y * eta) - T.log(1 + T.exp(y * eta)))
            # Data prior
            - T.sum(alpha * T.log(alpha / pi) +
                    (1 - alpha) * T.log(1 - alpha / (1 - pi)))
            # Variational entropy
            - .5 * T.sum(T.log(nu) + (eta - mu) * (eta - mu) / nu)
            # Variational prior
            - .5 * T.sum(alpha * (1 + T.log(tau) - T.log(gamma) -
                                 tau * alpha * (beta * beta + 1 / gamma)))
        )

        # Hyperparameter updates (Empirical Bayes)
        self.eb_updates = [(pi, T.sum(alpha) / self.shape[1]),
                           (tau, T.sum(alpha) / T.sum(alpha * alpha * (beta * beta + 1 / gamma)))]

    def fit(self, X_, y_, num_epochs=1000, a=1e-3, b1=0.9, b2=0.999, e=1e-3):
        if X_.shape != self.shape:
            raise ValueError("dimension mismatch: model {} vs X {}".format(self.shape, X_.shape))
        if X_.shape[0] != y_.shape[0]:
            raise ValueError("dimension mismatch: X {} vs y {}".format(X_.shape, y_.shape))

        # Stochastic gradient parameter updates (Adam)
        grad = T.grad(self.elbo, self.params)
        epoch = T.iscalar('epoch')
        M = [_S(_Z(param.get_value().shape), name='m{}'.format(param.name))
             for param in self.params]
        V = [_S(_Z(param.get_value().shape), name='v{}'.format(param.name))
             for param in self.params]
        a_t = a * T.sqrt(1 - b2 ** epoch) / (1 - b1 ** epoch)
        self.vb_updates = collections.OrderedDict()
        for param, g, m, v in zip(self.params, grad, M, V):
            new_m = b1 * m + (1 - b1) * g
            new_v = b2 * v + (1 - b2) * (g ** 2)
            self.vb_updates[param] = param + a_t * new_m / (T.sqrt(new_v) + e)
            self.vb_updates[m] = new_m
            self.vb_updates[v] = new_v

        # Observed data
        X = theano.shared(X_.astype(_real), name='X')
        y = theano.shared(y_, name='y')

        # Initial hyperparameters
        pi = theano.shared(numpy.array(1000 / X_.shape[1]).astype(_real), name='pi')
        tau = theano.shared(numpy.array(1).astype(_real), name='tau')

        givens = list(zip(self.obs, [X, y])) + list(zip(self.hyperparams, [pi, tau]))

        print('Compiling...', end='')
        _F = theano.function
        self.sg_step = _F([epoch], self.elbo, updates=self.vb_updates, givens=givens,
                          mode=theano.compile.MonitorMode(post_func=theano.compile.monitormode.detect_nan))
        self.eb_step = _F([], self.elbo, updates=self.eb_updates, givens=givens)
        print('done')

        print('Optimizing...', end='')
        # Optimize
        for epoch in range(num_epochs):
            elbo = self.sg_step(epoch + 1)
            elbo = self.eb_step()
        print('done')

        return elbo

def test():
    from .simulation import simulate
    theano.config.optimizer = 'fast_compile'
    x, y, theta = simulate(100, 100, .5)
    m = Model(*x.shape)
    m.fit(x, y)
