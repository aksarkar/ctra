"""Parsers for common formats

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import contextlib
import functools
import operator
import gzip

from .algorithms import *

def parse_oxstats(data):
    entries = (line.split() for line in data)
    for row in entries:
        row[2] = int(row[2])
        yield row

@contextlib.contextmanager
def oxstats_genotypes(sample_file, gen_file):
    """Return the list of samples and a generator which yields genotype
probabilities.

    Expects data in OXSTATS gen format (not bgen). This implementation does
    not allow random access to avoid memory issues.

    """
    with open(sample_file) as f:
        data = (line.split() for line in f)
        h1 = next(data)
        h2 = next(data)
        samples = list(data)
    with gzip.open(gen_file, 'rt') as f:
        yield h1, h2, samples, parse_oxstats(f)

def _merge_oxstats(seq1, seq2):
    """Merge lines of OXSTATS genotypes, matching on SNP attributes"""
    for a, b in join(seq1, seq2, key1=operator.itemgetter(2)):
        if a[:5] == b[:5]:
            yield a + b[5:]

def merge_oxstats(iterables):
    """Yield merged genotypes from iterables

    iterables - parsed OXSTATS data

    """
    for row in functools.reduce(_merge_oxstats, iterables):
        yield row

