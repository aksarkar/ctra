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

    def __init__(self, p, K, pve, min_maf=0.01, max_maf=0.5, m=None):
        self.p = p
        self.K = K
        self.pve = pve
        if m is None:
            self.m = p
            self.theta = R.normal(scale=numpy.sqrt(self.pve / self.m), size=self.m)
        else:
            self.m = m
            self.theta = numpy.zeros(self.p)
            self.theta[::self.p // self.m] = R.normal(size=self.m)
        self.maf = R.uniform(min_maf, max_maf, size=self.p)
        # Population mean and variance of genotype, according to the binomial
        # distribution
        self.x_mean = 2 * self.maf
        self.x_var = 2 * self.maf * (1 - self.maf)
        self.genetic_var = self.x_var * self.theta * self.theta
        self.residual_var = self.genetic_var.sum() * (1 / self.pve - 1)
        self.pheno_var = self.genetic_var.sum() + self.residual_var

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
