"""Simulate genotypes and phenotypes

We simulate ascertained case-control studies as follows:

1. Draw MAFs from Uniform(0.05, 0.5)
2. Pick m SNPs to be causal
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
pve), so the threshold can be computed beforehand as 1 - Phi(K). In our case,
we use the empirical variance of the liabilities to scale the Gaussian
CDF. This method assumes that the variance will be approximately constant for
large batch size.

After we have the desired number of control, rejection sampling cases is too
slow, so we adapt simCC from Golan et al, PNAS 2015 to sample the remainder of
the cases.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import pickle

import numpy
import numpy.random as R
import scipy.stats

def _multinomial(pmf):
    """Sample from multiple multinomial distributions in parallel. This is needed
to sample from individual-specific genotype conditional probabilities.

    Based on https://stackoverflow.com/a/34190035

    """
    cdf = pmf.cumsum(axis=0)
    query = R.rand(pmf.shape[1])
    return (cdf < query).sum(axis=0)

def sample_annotations(p):
    """Return a vector of annotations"""
    a = numpy.zeros(p)
    a[:p // 2] = 1
    return a

def sample_parameters(p, pve, m=None):
    """Return vector of MAFs, vector of effect sizes, and noise scale to achieve
desired PVE"""
    maf = R.uniform(0.05, 0.5, size=p)
    theta = numpy.zeros(p)
    if m is None:
        m = p // 10
    theta[::p // m] = R.normal(size=m)
    var_xtheta = numpy.sum(2 * maf * (1 - maf) * theta * theta)
    error_scale = numpy.sqrt(var_xtheta * (1 / pve - 1))
    liability_scale = numpy.sqrt(var_xtheta / pve)
    return maf, theta, error_scale, liability_scale

def sample_genotypes_iid(n, p, maf):
    """Return matrix of dosages.

    This implementation generates independent SNPs.

    """
    return R.binomial(2, maf, size=(n, p)).astype(float)

def sample_genotypes_given_case(n, p, maf, theta, liability_scale, t):
    """Return genotypes given all samples are cases.

    Based on algorithm "simCC" from Golan et al, PNAS 2015. We have to modify
    the algorithm when we relax the infinitesimal assumption. Specifically, we
    use the population variance of X (based on SNPs being independent with
    known MAF) to compute the residual variance needed to estimate p(y = 1 |
    g_{1..i}).

    """
    x = numpy.zeros(shape=(n, p))
    liabilities = numpy.zeros(n)
    var_xtheta = 2 * maf * (1 - maf) * theta * theta
    var_liability = liability_scale ** 2 - numpy.cumsum(var_xtheta)
    for i, (f, t, v) in enumerate(zip(maf, theta, var_liability)):
        prob_gi = numpy.array([f * f, 2 * f * (1 - f), (1 - f) * (1 - f)])
        prob_p_given_g = scipy.stats.norm(liabilities, v).sf(t)
        pmf = numpy.outer(prob_gi, prob_p_given_g)
        x[:,i] = _multinomial(pmf)
    return x

def compute_liabilities(x, theta, error_scale):
    """Return vector of liabilities"""
    n, p = x.shape
    return numpy.dot(x, theta) + R.normal(scale=error_scale, size=n)

def sample_case_control(n, p, K, P, pve, batch_size=1000, m=None):
    """Return genotypes and case-control phenotype with specified PVE

    K - case prevalence in the population
    P - target case proportion in the study

    """
    cases = numpy.zeros((1, p))
    controls = numpy.zeros((1, p))
    y = numpy.zeros(n)
    y[:int(n * P)] = 1
    maf, theta, error_scale, liability_scale = sample_parameters(p, pve, m)
    case_target = int(n * P)
    t = scipy.stats.norm(scale=liability_scale).isf(K)
    while n - case_target > 0:
        x = sample_genotypes_iid(batch_size, p, maf)
        l = compute_liabilities(x, theta, error_scale)
        case_index = l > t
        controls = numpy.vstack((controls, x[~case_index][:n - case_target]))
        num_taken_controls = x[~case_index][:n - case_target].shape[0]
        num_sampled_cases = x[case_index].shape[0]
        if case_target > 0 and num_sampled_cases > 0:
            cases = numpy.vstack((cases, x[case_index][:case_target]))
            num_taken_cases = x[case_index][:case_target].shape[0]
        else:
            num_taken_cases = 0
        case_target -= num_taken_cases
        n -= num_taken_cases + num_taken_controls
    additional_cases = sample_genotypes_given_case(case_target, p, maf, theta, liability_scale, t)
    x = numpy.vstack((cases[1:], additional_cases, controls[1:]))
    x -= x.mean(axis=0)[numpy.newaxis,:]
    return x, y, theta

def main(outfile, n=10000, p=10000, K=.01, P=.5, pve=.5, batch_size=1000, m=100):
    a = sample_annotations(p)
    x, y, theta = sample_case_control(n=n, p=p, K=K, P=P, pve=pve, batch_size=batch_size, m=m)
    with open(outfile, 'wb') as f:
        pickle.dump((x, y, a, theta), f)
