import os.path

import numpy
import numpy.random

random = numpy.random.RandomState(0)

def trial(model, m, n, n_causal, pve):
    freq = random.uniform(.01, .5)
    dosage = random.binomial(2, freq, (m, n))
    x = dosage - numpy.mean(dosage, axis=0)
    beta = numpy.zeros(m)
    beta[:n_causal] = random.normal(size=n_causal)
    g = numpy.dot(x.transpose(), beta)
    e = random.normal(size=n, scale=numpy.sqrt(numpy.var(g) * (1 / pve - 1)))
    y = g + e
    data = {'p': m,
            'n': n,
            'x': x.transpose(),
            'y': y}
    fit = model.sampling(data=data, pars=['pve'], seed=random, verbose=True)
    return fit

if __name__ == '__main__':
    with model(model_code=model_code) as model:
        print(trial(model, m=100, n=50, n_causal=10, pve=0.6))
