import numpy
import numpy.random as R

def simulate(n, p, pve):
    maf = R.uniform(0.05, 0.5, size=p)
    x = numpy.array([R.binomial(2, p, size=n) for p in maf])
    beta = numpy.zeros(p)
    m = p // 10
    beta[:m] = R.normal(size=m)
    g = x.dot(beta)
    e = R.normal(scale=numpy.var(g) * (1 / pve - 1))
    y = g + e
    return x, y, beta
