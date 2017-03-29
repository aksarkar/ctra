"""Algorithms needed throughout.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import itertools

import numpy

def join(seq1, seq2, key1, key2=None):
    """Yield pairs of elements s1, s2 in seq1, seq2 such that key1(s1) == key2(s2).

    This implementation performs a sort-merge join of two sequences. It
    requires the keys for both sequences to be comparable and for each sequence
    to be sorted on its key.

    seq1, seq2 - iterables
    key1 - Key for sequence 1
    key2 - Key for sequence 2. If None, assumed to be equal to key1

    """
    if key2 is None:
        key2 = key1
    seq1_buckets = itertools.groupby(seq1, key1)
    seq2_buckets = itertools.groupby(seq2, key2)
    k1, g1 = next(seq1_buckets)
    k2, g2 = next(seq2_buckets)
    while True:
        if k1 == k2:
            for pair in itertools.product(g1, g2):
                yield pair
            k1, g1 = next(seq1_buckets)
            k2, g2 = next(seq2_buckets)
        elif k1 < k2:
            k1, g1 = next(seq1_buckets)
        else:
            k2, g2 = next(seq2_buckets)

def kwise(iterable, k):
    it = iter(iterable)
    return zip(*[it for _ in range(k)])

def slice_sample(logp, init, num_samples=10000, warmup=5000):
    """Slice sampler (Neal, 2003)"""
    _U = numpy.random.uniform
    samples = numpy.zeros((num_samples, init.shape[0]))
    x = init
    fx = logp(init)
    for i in range(num_samples + warmup):
        fw = fx + numpy.log(_U())
        left = x.copy()
        right = x.copy()
        z = x.copy()
        for k in range(init.shape[0]):
            # Step out
            size = _U()
            left[k] -= size
            while logp(left) > fw:
                left[k] -= 0.25
            right[k] += 1 - size
            while logp(right) > fw:
                right[k] += 0.25

            # Step in
            fz = float('-inf')
            while fz <= fw:
                z[k] = left[k] + _U() * (right[k] - left[k])
                fz = logp(z)
                if fz > fw:
                    break
                elif z[k] > x[k]:
                    right[k] = z[k]
                elif z[k] < x[k]:
                    left[k] = z[k]
                else:
                    raise ValueError('Failed to step in')
        x = z
        fx = fz
        if i >= warmup:
            # In the multidimensional case, unpack the components of x
            samples[i - warmup] = x.ravel()
    return samples
