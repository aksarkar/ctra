"""Simulate genotypes and phenotypes

We simulate ascertained case-control studies as follows:

1. Draw MAFs from Uniform(0.05, 0.5)
2. Pick m = p / 10 SNPs to be causal
3. Draw causal effects theta from N(0, 1)
4. Sample dosages X for batches of 10,000 individuals from Binomial(2, MAF)
6. Draw noise e to achieve specified PVE in expectation from
   N(0, V[X theta] * (1 / pve - 1))
7. Compute liabilities X theta + e and liability threshold
   1 - Phi(K / sqrt(V[X theta + e])) per batch
8. Take cases and controls until achieving the desired study size

(7) is problematic because we do not make the infinitesimal assumption and
instead follow the Gaussian simulation scheme of Zhou et al, PLoS Genet 2013
(doi:10.1371/journal.pgen.1003264). Golan et al, PNAS 2015 derive the liability
threshold by simulating causal effects from N(0, pve) and noise from N(0, 1 -
pve), so the threshold can be computed beforehand as 1 - Phi(K).

In our case, we use the empirical variance of the liabilities to scale the
Gaussian CDF. This method assumes that the variance will be approximately
constant for large batch size.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import numpy
import numpy.random as R
import scipy.stats

def simulate_parameters(p, pve, m=None):
    """Return vector of MAFs, vector of effect sizes, and noise scale to achieve
desired PVE"""
    maf = R.uniform(0.05, 0.5, size=p)
    theta = numpy.zeros(p)
    if m is None:
        m = p // 10
    theta[:m] = R.normal(size=m)
    var_xtheta = numpy.sum(2 * maf * (1 - maf) * theta * theta)
    error_scale = numpy.sqrt(var_xtheta * (1 / pve - 1))
    liability_scale = numpy.sqrt(var_xtheta / pve)
    return maf, theta, error_scale, liability_scale

def simulate_genotypes(n, p, maf):
    """Return matrix of dosages.

    This implementation generates independent SNPs.

    """
    return R.binomial(2, maf, size=(n, p)).astype(float)

def simulate_liabilities(x, theta, error_scale):
    """Return vector of liabilities"""
    n, p = x.shape
    return numpy.dot(x, theta) + R.normal(scale=error_scale, size=n)

def simulate_gaussian(n, p, error_scale, center=True):
    """Return genotypes and Gaussian phenotype with specified PVE"""
    maf, theta, error_scale, _ = simulate_parameters(p)
    x = simulate_genotypes(n, p, maf)
    l = simulate_liabilities(x, theta, error_scale)
    if center:
        x -= x.mean(axis=0)[numpy.newaxis,:]
    return x, l, theta

def simulate_ascertained_probit(n, p, K, P, pve, batch_size=1000, center=True, m=None):
    """Return genotypes and case-control phenotype with specified PVE

    K - case prevalence in the population
    P - target case proportion in the study

    """
    cases = numpy.zeros((1, p))
    controls = numpy.zeros((1, p))
    y = numpy.zeros(n)
    y[:int(n * P)] = 1
    maf, theta, error_scale, liability_scale = simulate_parameters(p, pve, m)
    case_target = int(n * P)
    t = scipy.stats.norm(scale=liability_scale).isf(K)
    while case_target > 0 or n - case_target > 0:
        x = simulate_genotypes(batch_size, p, maf)
        l = simulate_liabilities(x, theta, error_scale)
        case_index = l > t
        num_sampled_cases = x[case_index].shape[0]
        if case_target > 0 and num_sampled_cases > 0:
            cases = numpy.vstack((cases, x[case_index][:case_target]))
            num_taken_cases = x[case_index][:case_target].shape[0]
        else:
            num_taken_cases = 0
        if n - case_target > 0:
            controls = numpy.vstack((controls, x[~case_index][:n - case_target]))
            num_taken_controls = x[~case_index][:n - case_target].shape[0]
        else:
            num_taken_controls = 0
        case_target -= num_taken_cases
        n -= num_taken_cases + num_taken_controls
    x = numpy.vstack((cases[1:], controls[1:]))
    if center:
        x -= x.mean(axis=0)[numpy.newaxis,:]
    return x, y, theta
