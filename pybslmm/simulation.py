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

_N = scipy.stats.norm()

def _multinomial(pmf):
    """Sample from multiple multinomial distributions in parallel. This is needed
to sample from individual-specific genotype conditional probabilities.

    Based on https://stackoverflow.com/a/34190035

    """
    cdf = pmf.cumsum(axis=0)
    query = R.rand(pmf.shape[1])
    return (cdf < query).sum(axis=0)

class Simulation():
    """Sample genotypes and phenotypes from a variety of genetic architectures."""

    def __init__(self, p, pve, min_maf=0.01, max_maf=0.5, seed=0):
        R.seed(seed)
        self.p = p
        self.pve = pve
        self.theta = R.normal(scale=numpy.sqrt(self.pve / self.p), size=self.p)
        self.maf = R.uniform(min_maf, max_maf, size=self.p)
        # Population mean and variance of genotype, according to the binomial
        # distribution
        self.x_mean = 2 * self.maf
        self.x_var = 2 * self.maf * (1 - self.maf)
        self.residual_var = 1 - self.pve

    def sample_annotations(self):
        """Return vector of annotations"""
        a = numpy.zeros(self.p, dtype='int32')
        a[:self.p // 2] = 1
        return a

    def sample_genotypes_iid(self, n):
        """Return matrix of dosages.

        This implementation generates independent SNPs, centered and scaled
        according to the population mean and population variance (based on
        MAF).

        """
        return (R.binomial(2, self.maf, size=(n, self.p)) - self.x_mean) / numpy.sqrt(self.x_var)

    def compute_liabilities(self, x):
        """Return vector of liabilities"""
        n, p = x.shape
        return x.dot(self.theta) + R.normal(scale=numpy.sqrt(self.residual_var), size=n)

    def sample_gaussian(self, n):
        """Return matrix of centered and scaled genotypes and vector of phenotypes"""
        x = self.sample_genotypes_iid(n)
        y = self.compute_liabilities(x)
        return x, y

    def sample_ascertained_probit(self, n, K, P, batch_size=1000):
        """Return matrix of centered and scaled genotypes and vector of phenotypes.

        This implementation uses rejection sampling to find samples with high
        enough liability to be cases.

        n - total samples
        K - population prevalence of cases
        P - target proportion of cases

        """
        case_target = int(n * P)
        remaining_cases = case_target
        x = numpy.zeros(shape=(n, self.p), dtype='float64')
        y = numpy.zeros(n, dtype='int32')
        y[:int(n * P)] = 1
        thresh = _N.isf(K)
        while remaining_cases > 0 or n - case_target > 0:
            z = self.sample_genotypes_iid(batch_size)
            l = self.compute_liabilities(z)
            case_index = l > thresh
            num_sampled_cases = z[case_index].shape[0]
            if remaining_cases > 0 and num_sampled_cases > 0:
                num_taken = min(remaining_cases, num_sampled_cases)
                x[remaining_cases - num_taken:remaining_cases] = z[case_index][:num_taken]
                remaining_cases -= num_taken
            if n - case_target > 0 and batch_size - num_sampled_cases > 0:
                num_taken = min(n - case_target, batch_size - num_sampled_cases)
                x[n - num_taken:n] = z[~case_index][:num_taken]
                n -= num_taken
        return x, y
