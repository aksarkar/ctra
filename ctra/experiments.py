import contextlib
import glob
import pickle
import os
import os.path

from matplotlib.pyplot import *
import numpy
import pandas
import scipy.special

switch_backend('pdf')

@contextlib.contextmanager
def pushdir(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    yield
    os.chdir(cwd)

def parse_results():
    files = glob.glob(os.path.join('[0-9]*.pkl'))
    results = []
    for f in files:
        with open(f, 'rb') as f:
            results.append(pandas.Series(pickle.load(f)))
    results = pandas.DataFrame(results)
    results['m0_b'] = [x[0] for x in results['m0_b']]
    results['true_b'] = [scipy.special.logit(numpy.sum([n for n, _ in a.annotation]) / a.num_variants)
                         for a in results['args']]
    return results

def plot_idealized_one_component(measure):
    if measure not in ('score', 'auprc'):
        raise ArgumentError
    results = parse_results()
    figure()
    results.boxplot(column='m0_b', by='true_b', grid=False,
                    return_type='axes')
    savefig('plot')
    close()
    figure()
    results.boxplot(column=[k for k in results.columns if measure in k],
                    by='true_b', grid=False)
    if measure == 'score':
        axhline(0.5, color='black')
    savefig('performance')
    close()

if __name__ == '__main__':
    with pushdir('gaussian-idealized-one-component'):
        plot_idealized_one_component('score')
    with pushdir('logistic-idealized-one-component'):
        plot_idealized_one_component('auprc')
