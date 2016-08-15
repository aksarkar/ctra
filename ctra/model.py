"""Fit the hierarchical model

We fit a generalized linear model regressing phenotype against genotype. We
impose a spike-and-slab prior on the coefficients to regularize the
problem. Our inference task is to estimate the posterior distribution of the
parameters pi (probability each SNP is causal) and tau (precision of causal
effects).

The inference requires integrating an intractable posterior over the
hyperparameters. Our strategy is to use importance sampling to perform the
integration, where the importance weights are the model evidence. We estimate
the importance weights by fitting a variational approximation to the
intractable posterior p(theta, z | x, y).

We cannot write an analytical solution for the variational approximation, so we
take a doubly stochastic approach, using Monte Carlo integration to estimate
intractable expectations (re-parameterizing integrals as sums) and drawing
samples (due to the non-conjugate prior) to estimate the gradient.

In our stochastic optimization, we use the sample mean of of the individual
sample likelihoods across the random samples eta as a control variate, since
its expectation is 0.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import drmaa
import pickle
import os
import sys

import numpy
import numpy.random as R
import scipy.misc
import scipy.special
import scipy.stats
import theano
import theano.tensor as T

import ctra.pcgc

_real = theano.config.floatX
_F = theano.function
_S = lambda x: theano.shared(x, borrow=True)
_Z = lambda n: numpy.zeros(n).astype(_real)

def logit(y, eta):
    """Return E_q[ln p(y | eta)] assuming a logit link."""
    F = y * eta - T.nnet.softplus(eta)
    return T.mean(T.sum(F, axis=1)) - T.mean(F)

def _grid(x, y=None):
    if y is None:
        y = x
    return numpy.dstack(numpy.meshgrid(x, y)).reshape(-1, 2)

def _clip(a, eps=1e-8):
    return numpy.clip(a, eps, 1 - eps)

class LogisticModel:
    """Class providing the implementation of the optimizer

    This is intended to provide a pickle-able object to re-use the Theano
    compiled function across hyperparameter samples.

    """
    def __init__(self, X_, y_, a_, llik=logit, minibatch_n=100,
                 max_precision=1e5, learning_rate=None, b1=0.9, b2=0.999,
                 e=1e-8):
        """Compile the Theano function which takes a gradient step

        llik - data likelihood under the variational approximation
        max_precision - maximum value of gamma
        learning_rate - initial gradient ascent step size (used for Adam)
        b1 - first moment exponential decay (Adam)
        b2 - second moment exponential decay (Adam)
        e - tolerance (Adam)

        """
        # Observed data
        X = _S(X_.astype(_real))
        y = _S(y_.astype(_real))
        a = _S(a_.astype('int8'))
        n, p = X_.shape
        self.m = 1 + max(a_)

        # Variational parameters
        alpha_raw = _S(_Z(p))
        alpha = T.nnet.sigmoid(alpha_raw)
        beta = _S(_Z(p))
        gamma_raw = _S(_Z(p))
        gamma = max_precision * T.nnet.sigmoid(gamma_raw)
        self.alpha = alpha
        self.beta = beta
        params = [alpha_raw, beta, gamma_raw]

        # Hyperparameters
        pi = T.vector()
        pi_deref = T.basic.choose(a, pi)
        tau = T.vector()
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

        # Maximize ELBO using stochastic gradient ascent
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
        self.vb_step = _F(inputs=[epoch, pi, tau], outputs=elbo,
                          updates=adam_updates)

        # Hyperparameter proposal distribution for Bayesian optimization using
        # random search
        self.logit_pi_proposal = scipy.stats.norm(loc=-3)
        self.tau_proposal = scipy.stats.lognorm(s=1)

    def sgvb(self):
        """Return optimum ELBO and variational parameters which achieve it

        Use Adaptive estimation to tune the learning rate and converge when
        change in ELBO changes sign. Poll ELBO every thousand iterations to
        reduce sensitivity to stochasticity.

        """
        pi = scipy.special.expit(self.logit_pi_proposal.rvs(size=self.m)).astype(_real)
        tau = self.tau_proposal.rvs(size=self.m).astype(_real)
        delta = 1
        t = 1
        curr_elbo = self.vb_step(t, pi, tau)
        while delta > 0:
            t += 1
            new_elbo = self.vb_step(t, pi, tau)
            if not t % 1000:
                delta = new_elbo - curr_elbo
                if delta > 0:
                    curr_elbo = new_elbo
        self.pi = pi
        self.tau = tau
        self.alpha = self.alpha.eval()
        self.beta = self.beta.get_value()

    def loss(X, y):
        """Return exponential loss of the model on data (X, y)"""
        return numpy.exp(y.dot(X.dot(alpha * beta))).sum()

def fit(X, y, a, tasks=10, eps=1e-8):
    """Return the posterior distributions P(theta, z | x, y, a) and P(pi, tau | x,
y, a)"""
    n, p = X.shape
    model = Model(X, y, a)
    with open('work.pkl', 'wb') as f:
        pickle.dump(model, f)
    with drmaa.Session() as s:
        j = s.createJobTemplate()
        j.remoteCommand = 'python'
        j.args = ['-m', 'ctra.model']
        ids = s.runBulkJobs(j, 1, tasks, 1)
        s.synchronize(ids)
    log_weights = numpy.zeros(shape=tasks)
    alpha = numpy.zeros(shape=(tasks, p))
    beta = numpy.zeros(shape=(tasks, p))
    for i in range(tasks):
        with open('{}.pkl'.format(i + 1), 'rb') as f:
            log_weights[i], alpha[i], beta[i] = pickle.load(f)
        log_weights[i] += logit_pi_prior.logpdf(scipy.special.logit(pi_)).sum()
        log_weights[i] += tau_prior.logpdf(tau_).sum()
    normalized_weights = numpy.exp(log_weights - scipy.misc.logsumexp(log_weights))
    return [normalized_weights.dot(x) for x in (alpha, beta, pi, tau)]

if __name__ == '__main__':
    task = int(os.environ['SGE_TASK_ID'])
    with open('work.pkl', 'rb') as f:
        model = pickle.load(f)
    alpha, beta, pi, tau = model.sgvb()
    with open('{}.pkl'.format(task), 'wb') as f:
        pickle.dump((alpha, beta, pi, tau), f)
