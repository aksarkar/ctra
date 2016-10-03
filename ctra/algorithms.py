"""Algorithms needed throughout.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import itertools

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
