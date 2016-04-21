import numpy
import numpy.random as R

def simulate(n, p, pve):
    maf = R.uniform(0.05, 0.5, size=p)
    x = R.binomial(2, maf, size=(n, maf.shape[0])).astype(float)
    x -= x.mean(axis=0)[numpy.newaxis,:]
    theta = numpy.zeros(p)
    m = p // 10
    theta[:m] = R.normal(size=m)
    g = numpy.dot(x, theta)
    e = R.normal(scale=numpy.var(g) * (1 / pve - 1), size=n)
    y = (g + e > 0).astype(int)
    return x, y, theta
