import matplotlib
import numpy
import theano
import theano.tensor as T
from matplotlib.pyplot import *

switch_backend('pdf')

x = numpy.random.normal(size=100)
e = 2 * numpy.random.normal(size=100)
y = x + e

X = theano.shared(x)
Y = theano.shared(y)
n = y.shape[0]

mu_mean = theano.shared(numpy.zeros(x.shape), name='mu_mean')
mu_log_prec = theano.shared(numpy.ones(x.shape), name='mu_log_prec')
mu_prec = 1e-3 + T.nnet.softplus(mu_log_prec)

log_sigma2_mean = theano.shared(0., name='log_sigma2_mean')
log_sigma2_log_prec = theano.shared(1., name='log_sigma2_log_prec')
log_sigma2_prec = 1e-3 + T.nnet.softplus(log_sigma2_log_prec)

mu_prior_mean = numpy.zeros(x.shape)
mu_prior_prec = numpy.ones(x.shape)
log_sigma2_prior_mean = 0
log_sigma2_prior_prec = 1

noise = theano.shared(numpy.random.normal(size=100).reshape(-1, 1))
epoch = T.iscalar()

eta_batch = (epoch + 1) % 10
eta_raw = noise[eta_batch * 10:(eta_batch + 1) * 10]
eta = mu_mean + T.addbroadcast(eta_raw, 1) * T.sqrt(1 / mu_log_prec)

batch = epoch % 10
phi_raw = noise[batch * 10:(batch + 1) * 10]
phi = T.exp(T.addbroadcast(log_sigma2_mean + phi_raw * T.sqrt(1 / log_sigma2_prec), 1))

llik = -.5 * T.mean(T.sum(T.log(phi) + T.sqr(Y - eta) / phi, axis=1))
kl_mu = .5 * T.sum(1 + T.log(mu_prior_prec) - T.log(mu_prec) + mu_prior_prec * (T.sqr(mu_mean - mu_prior_mean) + 1 / mu_prior_prec))
kl_sigma2 = .5 * T.sum(1 + T.log(log_sigma2_prior_prec) - T.log(log_sigma2_prec) + log_sigma2_prior_prec * (T.sqr(log_sigma2_mean - log_sigma2_prior_mean) + 1 / log_sigma2_prior_prec))
elbo = llik - kl_mu - kl_sigma2
params = [mu_mean, mu_log_prec, log_sigma2_mean, log_sigma2_log_prec]
grad = T.grad(elbo, params)
step = theano.function(inputs=[epoch], outputs=[llik, kl_mu, kl_sigma2, T.sqr(mu_mean - X).sum()] + params, updates=[(p, p + 1e-3 * g) for p, g in zip(params, grad)])

nsteps = 2000
trace = [step(t) for t in range(nsteps)]

fig, axes = subplots(3, 1)
fig.set_size_inches(9, 12)
axes[0].plot(numpy.arange(nsteps), [x[:3] for x in trace])
axes[0].legend(['Neg. error', 'KL(mu)', 'KL(sigma2)'])
axes[1].plot(numpy.arange(nsteps), [x[-2] for x in trace])
axes[1].axhline(numpy.log(2), color='black')
axes[1].legend(['log(sigma2)'])
axes[2].plot(numpy.arange(nsteps), [x[3] for x in trace])
axes[2].legend(['Loss'])
savefig('llik.pdf')
close()
