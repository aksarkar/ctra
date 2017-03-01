import argparse
import contextlib
import itertools
import logging
import os
import sys

import h5py
import numpy

import ctra.algorithms
import ctra.formats

logger = logging.getLogger(__name__)
_A = argparse.ArgumentTypeError

# Constants
_alleles = numpy.arange(3)
_sq_alleles = numpy.square(_alleles)

def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', '--load-oxstats', nargs='+', help='OXSTATS data sets (.sample, .gen.gz)', default=None)
    parser.add_argument('-n', '--num-samples', type=int, help='Number of samples', required=True)
    parser.add_argument('-p', '--num-variants', type=int, help='Number of genetic variants', required=True)
    parser.add_argument('-o', '--out', help='Output file', required=True)
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite')
    parser.add_argument('--chunk-size', help='HDF5 chunk size', default=(100, 100))
    parser.add_argument('-l', '--log-level', choices=['INFO', 'DEBUG'], help='Log level', default='INFO')
    return parser

def info(probs):
    """Compute the ratio between observed and complete information.

    This implementation follows the description in
    https://mathgen.stats.ox.ac.uk/genetics_software/snptest/snptest.v2.pdf

    """
    e = probs.reshape(-1, 3).dot(_alleles)
    f = probs.reshape(-1, 3).dot(_sq_alleles)
    theta_hat = e.sum() / (2 * len(e))
    info = 1
    if theta_hat > 0 and theta_hat < 1:
        info -= (f - numpy.square(e)).sum() / (2 * len(e) * theta_hat * (1 - theta_hat))
    return e, info

def convert():
    """Entry point for format conversion"""
    parser = _parser()
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    with contextlib.ExitStack() as stack:
        data = [stack.enter_context(ctra.formats.oxstats_genotypes(*a))
                for a in ctra.algorithms.kwise(args.load_oxstats, 2)]
        samples = list(itertools.chain.from_iterable(s for _, _, s, _ in data))
        merged = ctra.formats.merge_oxstats([d for _, _, _, d in data])
        if args.num_samples > len(samples):
            logger.error('{} individuals present in OXSTATS data, but {} were specified'.format(len(samples), args.num_samples))
            sys.exit(1)
        elif args.num_samples < len(samples):
            logger.warn('{} individuals present in OXSTATS data, but {} were specified'.format(len(samples), args.num_samples))
        if os.path.exists(args.out) and not args.force:
            logger.error('Output file {} already exists. Not overwriting')
            sys.exit(1)
        outfile = stack.enter_context(h5py.File(args.out, 'w'))
        outfile.create_dataset('dosage', shape=(args.num_samples, args.num_variants), dtype='float32', chunks=args.chunk_size)
        outfile.create_dataset('info', shape=(1, args.num_variants), dtype='float32')
        for j, row in enumerate(merged):
            if j >= args.num_variants:
                logger.warn('{} variants processed, but additional variants are present'.format(j))
                break
            probs = numpy.array([float(x) for x in row[5:]])
            x, y = info(probs)
            outfile['dosage'][:, j] = x
            if not j % 1000:
                logger.debug('{} variants processed'.format(j))
        if j + 1 < args.num_variants:
            logger.error('{} variants present in OXSTATS data, but {} were specified'.format(j, args.num_variants))
            sys.exit(1)
