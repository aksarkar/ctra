"""Simulate genotypes and phenotypes

The basic generative model is based on Zhou et al, PLoS Genet 2013 and de los
Campos et al. PLoS Genet 2015. The outline of the generative model is:

1. Sample minor allele frequencies f ~ U(min_maf, max_maf)
2. (Optional) Partition the genome into annotations
3. Sample effects b ~ N(0, scale) according to annotation
4. Compute genetic variance as V[X b] = \sum_j 2 f_j (1 - f_j) b_j^2 and
   remaining variance components from target PVE
5. Sample dosages i.i.d. x_i ~ Binomial(2, f)

The key features of this model are:

1. Dosages are not normalized to variance one, and the population variance of
   dosages is included in the genetic variance. This is opposed to normalizing
   dosages to variance one so they drop out of the genetic variance.
2. The actual genetic variance (based on MAF and effect size) is used
   to determine the residual variance. This is opposed to using the expected
   genetic variance (in the one-component case, the PVE by construction).

We additionally implement the simCC algorithm from Golan et al, PNAS 2015 to
quickly sample ascertained case-control genotypes. The basic idea is to sample
from the conditional distribution of remaining genotypes given phenotype and
the current genotypes:

p(x_1, ..., x_j | y) \propto p(y | x_1, ..., x_j) p(x_1, ..., x_j)
p(x_1, ..., x_j) = p(x_j | x_1, ..., x_{j-1})
p(l | x_1, ..., x_j) = N(x_{1..j} b, V[y] - V[x_{j..p} b])
p(y | l) = p(l < t), y = 0; p(l > t), y = 1

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import logging
import contextlib
import pickle
import os.path

import numpy
import scipy.stats

R = numpy.random
_N = scipy.stats.norm()

logger = logging.getLogger(__name__)

def _multinomial(pmf):
    """Sample from multiple multinomial distributions in parallel. This is needed
to sample from individual-specific genotype conditional probabilities.

    Based on https://stackoverflow.com/a/34190035

    """
    cdf = pmf.cumsum(axis=0)
    query = R.rand(pmf.shape[1])
    return (cdf < query).sum(axis=0)

class Simulation:
    """Sample genotypes and phenotypes from a variety of genetic architectures."""

    def __init__(self, p, min_maf=0.01, max_maf=0.5, seed=0):
        """Initialize the SNP parameters of the simulation.

        By default all SNPs are given the same annotation.

        p - total number of SNPs
        min_maf - minimum MAF of SNPs
        max_maf - maximum MAF of SNPs
        seed - random seed

        """
        logging.info('Initializing simulation')
        R.seed(seed)
        self.p = p
        self.maf = R.uniform(min_maf, max_maf, size=self.p)
        # Population mean and variance of genotype, according to the binomial
        # distribution
        self.x_mean = 2 * self.maf
        self.x_var = 2 * self.maf * (1 - self.maf)
        # Should raise an exception if this is used before sampling
        self.theta = None
        # By default, put all SNPs in the same annotation
        self.num_annot = 0
        # Keep the right-most index of each annotation
        self.annot_index = numpy.array([self.p])
        self.annot = numpy.zeros(self.p, dtype='int32')

    def _annotations(self):
        """Yield blocks of annotated SNPs"""
        start = 0
        for end in self.annot_index:
            yield start, end
            start = end

    def sample_annotations(self, proportion=None):
        """Generate annotations covering target proportion of SNPs.

        proportion - (num_annotations,)

        """
        if proportion is None:
            proportion = numpy.array([1])
        self.annot_index = numpy.cumsum(self.p * proportion).astype('int32')
        for i, (start, end) in enumerate(self._annotations()):
            self.annot[start:end] = i
        return self

    def sample_effects(self, pve, annotation_params=None):
        """Generate SNP effects according to annotations.

        Generate residual variance and phenotypic variance based on target
        PVE. By default all SNPs are given the same annotation and effect size
        variance one.

        pve - total PVE
        annotation_params - list of tuples (number of causal variants, population variance of effect size)

        """
        if not 0 < pve < 1:
            raise ValueError('PVE must be in (0, 1)')
        elif annotation_params is not None and len(annotation_params) != self.annot_index.shape[0]:
            raise ValueError('Shape mismatch: annotations parameters {} vs. annotations {}'.format(len(annotation_params), self.annot_index.shape[0]))
        if annotation_params is None:
            annotation_params = [(end - start, 1) for start, end
                                 in self._annotations()]
        elif any(p[0] < 0 for p in annotation_params):
            raise ValueError('Number of causal variants must be non-negative')
        elif any(p[1] <= 0 for p in annotation_params):
            raise ValueError('Effect size variance must be non-negative')

        logging.info('Sampling SNP parameters')
        self.theta = numpy.zeros(self.p)
        for (num, scale), (start, end) in zip(annotation_params, self._annotations()):
            self.theta[start:end][:num] = R.normal(scale=scale, size=num)

        # Keep genetic variance as a vector (per-SNP) for quicker case-control
        # sampling: at each step we need the remaining variance. Don't ignore
        # population variance of genotype in computing the genetic variance
        self.genetic_var = self.x_var * numpy.square(self.theta)
        self.pheno_var = self.genetic_var.sum() / pve
        self.residual_var = self.pheno_var - self.genetic_var.sum()
        return self

    def sample_genotypes_iid(self, n):
        """Return matrix of dosages.

        This implementation generates SNPs in linkage equilibrium
        (i.i.d). Dosages are centered about the population mean (based on MAF)
        but not scaled. We keep the variance of genotype in the generative
        model, following de los Campos et al. PLoS Genet 2015.

        """
        x = R.binomial(2, self.maf, size=(n, self.p)) - self.x_mean
        return x

    def compute_liabilities(self, x, normalize=False):
        """Return normalized vector of liabilities"""
        if self.theta is None:
            raise ValueError('Need to sample theta first')
        genetic_value = x.dot(self.theta)
        genetic_value += numpy.sqrt(self.residual_var) * R.normal(size=x.shape[0])
        if normalize:
            genetic_value /= numpy.sqrt(self.pheno_var)
        return genetic_value

    def sample_gaussian(self, n):
        """Return matrix of genotypes and vector of phenotypes"""
        x = self.sample_genotypes_iid(n)
        y = self.compute_liabilities(x)
        return x, y

    def sample_ascertained_probit(self, n, K, P, batch_size=1000):
        """Return matrix of genotypes and vector of phenotypes.

        This implementation uses rejection sampling to find samples with high
        enough liability to be cases.

        n - total samples
        K - population prevalence of cases
        P - target proportion of cases

        """
        logging.info('Sampling individuals conditioned on phenotype')
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

    def sample_genotypes_given_pheno(self, n, K, case=True):
        """Sample matrix of genotypes given phenotype.

        Based on algorithm "simCC" from Golan et al, PNAS 2015. We have to modify
        the algorithm when we relax the infinitesimal assumption. Specifically, we
        use the population variance of X (based on SNPs being independent with
        known MAF) to compute the residual variance needed to estimate p(y = 1 |
        g_{1..i}).

        """
        if self.theta is None:
            raise ValueError('Need to sample theta first')
        x = numpy.zeros(shape=(n, self.p))
        y = numpy.zeros(n)
        thresh = numpy.sqrt(self.pheno_var) * _N.isf(K)
        remaining_pheno_scale = numpy.sqrt(self.pheno_var - numpy.cumsum(self.genetic_var))
        z = (numpy.arange(3)[:, numpy.newaxis] - self.x_mean).T
        prob_z = numpy.column_stack((numpy.square(1 - self.maf),
                                     2 * self.maf * (1 - self.maf),
                                     numpy.square(self.maf)))
        for j in range(self.p):
            new_y = y[:, numpy.newaxis] + z[j] * self.theta[j]
            prob_p_given_g = _N.sf((thresh - new_y.T) / remaining_pheno_scale[j])
            if not case:
                prob_p_given_g = 1 - prob_p_given_g
            pmf = prob_p_given_g * prob_z[j][:, numpy.newaxis]
            pmf /= pmf.sum(axis=0)
            x_j = _multinomial(pmf)
            x[:, j] = x_j
            y = new_y[numpy.arange(n), x_j]
        return x

    def sample_case_control(self, n, K, P, batch_size=1000):
        """Return matrix of genotypes and vector of phenotypes.

        This implementation uses a faster algorithm to sample from the
        conditional distribution of genotypes given phenotype.

        n - total samples
        P - target proportion of cases

        """
        case_target = int(n * P)
        x = numpy.zeros((n, self.p), dtype='float32')
        y = numpy.zeros(n, dtype='int32')
        y[:case_target] = 1
        thresh = _N.isf(K)
        while n > case_target:
            samples = min(n - case_target, batch_size)
            x[n - samples:n] = self.sample_genotypes_given_pheno(samples, K, False)
            n -= samples
        while case_target > 0:
            samples = min(case_target, batch_size)
            x[case_target - samples:case_target] = self.sample_genotypes_given_pheno(samples, K, True)
            case_target -= samples
        return x, y

@contextlib.contextmanager
def simulation(p, pve, annotation_params, seed):
    """Context manager to cache the simulation generative model.

    If a simulation with the specified hyperparameters is not cached, sample
    the remaining parameters and write the resulting model out.

    The genome is divided into two equal-sized annotations by default.

    """
    key = 'simulation-{}-{}-{}-{}.pkl'.format(p, pve, annotation_params, seed)
    if not os.path.exists(key):
        s = Simulation(p=p, seed=seed)
        s.sample_annotations(proportion=numpy.ones(len(annotation_params)) / len(annotation_params))
        s.sample_effects(pve=pve, annotation_params=annotation_params)
        with open(key, 'wb') as f:
            pickle.dump(s, f)
        yield s
    else:
        with open(key, 'rb') as f:
            yield pickle.load(f)
