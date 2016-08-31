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

def estimate(y, grm, K=None):
    """Naive estimate of PVE (without fixed effects).

    This provides a nonstreaming implementation as a sanity check. If K is
    None, assume y is Gaussian and do not perform correction for ascertainment.

    y - phenotype vector
    grm - (k, n * (n - 1) / 2, 1) ndarray of upper-triangular GRM entries
    K - prevalence of case-control phenotype

    """
    y = numpy.copy(y)
    y -= y.mean()
    y /= y.std()
    index = numpy.triu_indices(y.shape[0], 1)
    if K is None:
        c = 1
        prm = numpy.outer(y, y)[index].reshape(-1, 1)
    else:
        t = _N.isf(K)
        z = _N.pdf(t)
        P = numpy.mean(y)
        c = K ** 2 * (1 - K) ** 2 / (z ** 2 * P * (1 - P))
        prm = numpy.outer(y - P, y - P)[index].reshape(-1, 1) / (P * (1 - P))
    return c * scipy.linalg.lstsq(grm, prm)[0]

def _estimate_grm(x):
    """Return upper triangular entries of the GRM estimated from x"""
    x -= numpy.mean(x, axis=0)
    x /= numpy.std(x, axis=0)
    grm = numpy.inner(x, x)[numpy.triu_indices(x.shape[0], 1)].reshape(-1, 1)
    grm /= x.shape[1]
    return grm

def grm(x, a):
    """Return matrix of GRM entries per partition of x

    x - dosage matrix
    a - SNP annotations

    """
    return numpy.column_stack([_estimate_grm(x[:,a == i]) for i in range(1 + max(a))])
