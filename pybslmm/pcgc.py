"""Implement PCGC regression for heritability estimation

Provides a streamining implementation based on:

- Bhatia, G., Gusev, A., Loh, P., Vilhj\'almsson, Bjarni J, Ripke, S., Purcell,
  S., Stahl, E., Daly, M., de Candia, T. R., Kendler, K. S., O'Donovan, M. C.,
  Lee, S. H., Wray, N. R., Neale, B. M., Keller, M. C., Zaitlen, N. A.,
  Pasaniuc, B., Yang, J., Price, A. L., .. (2015). Haplotypes of common snps
  can explain missing heritability of complex diseases. bioRxiv, (), .
  http://dx.doi.org/10.1101/022418

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

def _logit(X, beta):
    """Logit link function"""
    return exp(dot(X, beta)) / (1.0 + exp(dot(X, beta)))

def _logit_ll(beta, X, y):
    """Logit regression negative log likelihood"""
    return -numpy.sum(y * dot(X, beta) - log(1.0 + exp(dot(X, beta))))

def _logit_score(beta, X, y):
    """Logit regression score matrix (Jacobian of the negative log likelihood)"""
    return dot(X.T, _logit(X, beta) - y)

def logit_regression(X, y):
    """Return maximum likelihood estimate of beta for the model:

    logit(p) = X \beta

    http://stackoverflow.com/a/13898232

    """
    opt = scipy.optimize.minimize(_logit_ll, numpy.zeros(X.shape[1]),
                                  args=(X, y), method='BFGS',
                                  jac=_logit_score)
    return opt

def estimate_thresholds(phenotype, fixed_effects, prevalence):
    """Return ascertainment-corrected conditional probability of being a case,
fixed effect-corrected liability, thresholds, and total phenotypic variance

    """
    fit = logit_regression(fixed_effects, phenotype)
    if not fit.success:
        raise RuntimeError('Failed to regress phenotype on fixed effects')
    ascertainment = numpy.mean(phenotype)
    correction = (1 - ascertainment) / ascertainment * prevalence / (1 - prevalence)
    ascertained_probs = fixed_effects.dot(fit.x)
    probs = correction * ascertained_probs / (1 + correction * ascertained_probs - ascertained_probs)

    thresholds = _N.sf(probs)
    cases = thresholds[numpy.all([phenotype == 1, numpy.isfinite(thresholds)])]
    controls = thresholds[numpy.all([phenotype == 0, numpy.isfinite(thresholds)])]
    pheno_var = (prevalence * numpy.var(cases) +
                 (1 - prevalence) * numpy.var(controls) +
                 prevalence * (1 - prevalence) * (numpy.mean(cases) - numpy.mean(controls)) ** 2)

    mult = (ascertainment - prevalence) / (ascertainment * (1 - prevalence))
    mult = 1 - (ascertained_probs * mult)
    mult *= _N.pdf(thresholds)
    mult /= (probs * (1 - correction) + correction )

    return probs, mult, pheno_var

def _estimate(pheno_var, numerator, denominator):
    """Return estimates of variance components"""
    print(denominator)
    coeff = scipy.linalg.inv(denominator).dot(numerator) / (1 + pheno_var)

def _jacknife(pheno_var, numerator, denominator):
    """Return jacknife standard errors of variance components"""
    return numpy.std(_estimate(pheno_var,
                               sum(n for j, n in enumerate(numerator) if not j % i),
                               sum(d for j, d in enumerate(denominator) if not j % i))
                     for i in range(len(numerator)))

def estimate(y, grm):
    """Naive estimate of PVE (without fixed effects).

    This provides a nonstreaming implementation as a sanity check.

    """
    index = numpy.triu_indices(grm.shape[0], 1)
    if len(grm.shape) == 2:
        G = grm[index].reshape(-1, 1)
    else:
        raise ValueError('Incorrect dimension of GRM: {}'.format(grm.shape))
    prm = numpy.outer(y, y)[index].reshape(-1, 1)
    return scipy.linalg.lstsq(G, prm)[0]

def grm(x):
    """Return the GRM estimated from SNPs in x"""
    w = x - numpy.mean(x, axis=0)
    w /= numpy.std(w, axis=0)
    return numpy.inner(w, w) / w.shape[1]

def partitioned_grm(x, a):
    """Return the GRMs estimated on partitions of X, according to annotation a"""
    return numpy.array([grm(x[:,a == i]) for i in range(1 + max(a))])

if __name__ == '__main__':
    import pickle
    import sys

    with open(sys.argv[1], 'rb') as f:
        x, y, a, theta = pickle.load(f)
    n, p = x.shape
    a = a.astype('int32')
    pve = _naive_estimate(y, _partitioned_grm(x, a), 0.01)
    print(pve)
