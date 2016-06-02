"""Fit the hierarchical model

We fit a generalized linear model regressing phenotype against genotype. We
impose a spike-and-slab prior on the coefficients to regularize the
problem. Our inference task is to estimate the posterior distribution of the
parameters pi (probability each SNP is causal) and tau (precision of causal
effects).

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

import numpy
import numpy.random as R
import scipy.special
import theano
import theano.tensor as T

_real = theano.config.floatX
_F = theano.function
_S = theano.shared
_Z = lambda n: numpy.zeros(n).astype(_real)

def logit(y, eta):
    """Return E_q[ln p(y | eta)] assuming a logit link."""
    F = y * eta - T.nnet.softplus(eta)
    return T.mean(T.sum(F, axis=1)) - T.mean(F)

def fit(X_, y_, a_, pi_, tau_, llik=logit, initial_params=None,
        minibatch_n=100, max_precision=1e5, learning_rate=None,
        b1=0.9, b2=0.999, e=1e-8):
    """Return the variational parameters alpha, beta, gamma.

    X_ - dosage matrix (n x p)
    y_ - response vector (n x 1)
    a_ - annotation vector (p x 1), values {0, ..., m - 1}
    llik - data likelihood under the variational approximation
    initial_params - Initial values of (alpha, beta, gamma, pi, tau)
    minibatch_n - size of sample minibatches
    max_precision - maximum value of gamma
    learning_rate - initial gradient ascent step size (used for Adam)
    b1 - first moment exponential decay (Adam)
    b2 - second moment exponential decay (Adam)
    e - tolerance (Adam)

    """
    if X_.shape[0] != y_.shape[0]:
        raise ValueError("dimension mismatch: X {} vs y {}".format(X_.shape, y_.shape))
    if X_.shape[1] != a_.shape[0]:
        raise ValueError("dimension mismatch: X {} vs a {}".format(X_.shape, a_.shape))
    n, p = X_.shape
    m = 1 + max(a_)

    # Observed data
    X = theano.shared(X_.astype(_real))
    y_obs = theano.shared(y_)
    y = T.cast(y_obs, 'int32')
    a = theano.shared(a_.astype('int32'))

    # Randomly initialize over hyperparameter proposals to find optimal
    # initialization, then re-run with that initialization for each proposal
    if initial_params is None:
        alpha_ = R.uniform(size=p).astype(_real)
        beta_ = R.normal(size=p).astype(_real)
        gamma_ = R.normal(size=p).astype(_real)

    # Variational parameters
    alpha_raw = _S(alpha_)
    alpha = T.nnet.sigmoid(alpha_raw)
    beta = _S(beta_)
    gamma_raw = _S(gamma_)
    gamma = max_precision * T.nnet.sigmoid(gamma_raw)
    params = [alpha_raw, beta, gamma_raw]

    # Hyperparameters
    pi = _S(pi_.astype(_real))
    pi_deref = T.basic.choose(a, pi)
    tau = _S(tau_.astype(_real))
    tau_deref = T.basic.choose(a, tau)

    # We need to perform inference on minibatches of samples for speed. Rather
    # than taking balanced subsamples, we take a sliding window over a
    # permutation which is balanced in expectation.
    perm = _S(R.permutation(n).astype('int32'))
    epoch = T.iscalar()
    sample_minibatch = epoch % (n // minibatch_n)
    index = perm[sample_minibatch * minibatch_n:(sample_minibatch + 1) * minibatch_n]
    X_s = X[index]
    y_s = y[index]

    # Variational approximation (re-parameterize eta = X theta). This is a
    # "Gaussian reconstruction" in that we characterize its expectation and
    # variance, then approximate its distribution with a Gaussian.
    #
    # We need to take the gradient of an intractable integral, so we re-write
    # it as a Monte Carlo integral which is differentiable, following Kingma &
    # Welling, ICLR 2014 (http://arxiv.org/abs/1312.6114). When we take the
    # gradient, the global mean of the reconstruction is constant and drops
    # out, so we only need to keep the global variance.
    mu = T.dot(X_s, alpha * beta)
    nu = T.dot(T.sqr(X_s), alpha / gamma + alpha * (1 - alpha) + T.sqr(beta))
    random = T.shared_randomstreams.RandomStreams(seed=0)
    eta_raw = random.normal(size=(10, minibatch_n))
    eta = mu + T.sqrt(nu) * eta_raw

    # Objective function
    elbo = (
        llik(y_s, eta)
        + .5 * T.sum(alpha * (1 + T.log(tau_deref) - T.log(gamma) - tau_deref * (T.sqr(beta) + 1 / gamma)))
        - T.sum(alpha * T.log(alpha / pi_deref) + (1 - alpha) * T.log((1 - alpha) / (1 - pi_deref)))
    )

    # Maximize ELBO using stochastic gradient descent.
    #
    # Adaptive estimation (Kingma & Welling arxiv:1412.6980) tunes the learning
    # rate based on exponentially weighted moving averages of the first and
    # second moments of the gradient.
    if learning_rate is None:
        learning_rate = 0.5 / n
    grad = T.grad(elbo, params)
    M = [_S(_Z(p)) for param in params]
    V = [_S(_Z(p)) for param in params]
    a_t = learning_rate * T.sqrt(1 - T.pow(b2, epoch)) / (1 - T.pow(b1, epoch))
    adam_updates = collections.OrderedDict()
    for p, g, m, v in zip(params, grad, M, V):
        new_m = b1 * m + (1 - b1) * g
        new_v = b2 * v + (1 - b2) * T.sqr(g)
        adam_updates[p] = T.cast(p + a_t * new_m / (T.sqrt(new_v) + e), _real)
        adam_updates[m] = new_m
        adam_updates[v] = new_v
    vb_step = _F(inputs=[epoch], outputs=elbo, updates=adam_updates)
    curr_elbo = numpy.array(0)
    delta = 1
    t = 0
    while delta > 0:
        t += 1
        new_elbo = vb_step(t)
        if not t % 1000:
            delta = new_elbo - curr_elbo
            curr_elbo = new_elbo
            yield curr_elbo, alpha.eval(), beta.get_value(), gamma.eval()
