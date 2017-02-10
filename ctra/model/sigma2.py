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
kl = .5 * T.sum(1 + T.log(mu_prior_prec) - T.log(mu_prec) + mu_prior_prec * (T.sqr(mu_mean - mu_prior_mean) + 1 / mu_prior_prec))
kl += .5 * T.sum(1 + T.log(log_sigma2_prior_prec) - T.log(log_sigma2_prec) + log_sigma2_prior_prec * (T.sqr(log_sigma2_mean - log_sigma2_prior_mean) + 1 / log_sigma2_prior_prec))
elbo = llik - kl

params = [mu_mean, mu_log_prec, log_sigma2_mean, log_sigma2_log_prec]
grad = T.grad(elbo, params)
step = theano.function(inputs=[epoch], outputs=[elbo, llik, kl, log_sigma2_mean], updates=[(p, p + 1e-3 * g) for p, g in zip(params, grad)])
trace = numpy.array([step(t) for t in range(1000)])

fig, axes = subplots(2, 1)
axes[0].plot(numpy.arange(1000), trace[:,:3])
axes[0].legend(['ELBO', 'Neg. error', 'KL'])
axes[1].plot(numpy.arange(1000), trace[:,3])
axes[1].axhline(2)
axes[1].set_title('log(sigma2)')
savefig('llik.pdf')
close()
