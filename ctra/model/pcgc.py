"""Implement PCGC regression for heritability estimation

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import numpy
import scipy.linalg
import scipy.optimize
import scipy.stats

exp = numpy.exp
log = numpy.log
dot = numpy.dot

_N = scipy.stats.norm()

def pcgc(y, grm, K=None):
    """Naive estimate of PVE (without fixed effects, without standard error).

    If K is None, assume y is Gaussian and do not perform correction for
    ascertainment.

    y - phenotype vector
    grm - (k, n * (n - 1) / 2, 1) ndarray of upper-triangular GRM entries
    K - prevalence of case-control phenotype

    """
    index = numpy.triu_indices(y.shape[0], 1)
    if K is None:
        y = numpy.copy(y)
        y -= y.mean()
        y /= y.std()
        c = 1
        prm = numpy.outer(y, y)[index].reshape(-1, 1)
    else:
        t = _N.isf(K)
        z = _N.pdf(t)
        P = numpy.mean(y)
        c = K ** 2 * (1 - K) ** 2 / (z ** 2 * P * (1 - P))
        prm = numpy.outer(y - P, y - P)[index].reshape(-1, 1) / (P * (1 - P))
    return c * scipy.linalg.lstsq(grm, prm)[0].ravel()

def _estimate_grm(x):
    """Return upper triangular entries of the GRM estimated from x"""
    x -= numpy.mean(x, axis=0)
    x /= (numpy.std(x, axis=0) + 1e-6)
    grm = numpy.inner(x, x)[numpy.triu_indices(x.shape[0], 1)].reshape(-1, 1)
    grm /= x.shape[1]
    return grm

def grm(x, a):
    """Return matrix of GRM entries per partition of x

    x - dosage matrix
    a - SNP annotations

    """
    return numpy.column_stack([_estimate_grm(x[:,a == i]) for i in range(1 + max(a))])
