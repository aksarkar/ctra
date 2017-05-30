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

def parse_results(measure):
    files = glob.glob(os.path.join('[0-9]*.pkl'))
    estimated_prop = []
    performance = []
    perf_columns = None
    for f in files:
        with open(f, 'rb') as f:
            result = pickle.load(f)
            num_causal = len(numpy.where(result['simulation'].theta != 0)[0])
            num_variants = result['simulation'].p
            estimated_prop.append([scipy.special.logit(num_causal / num_variants), result['m0_b'][0]])
            if perf_columns is None:
                perf_columns = [k for k in result if measure in k]
            performance.append([scipy.special.logit(num_causal / num_variants)] +
                               [result[k] for k in perf_columns])
    estimated_prop = pandas.DataFrame(estimated_prop)
    estimated_prop.columns = ['true_pi', 'estimated_pi']
    performance = pandas.DataFrame(performance)
    performance.columns = ['true_pi'] + perf_columns
    return estimated_prop, performance

def plot_idealized_one_component(measure='score'):
    if measure not in ('score', 'auprc'):
        raise ArgumentError
    estimated_prop, performance = parse_results(measure)
    figure()
    ax = estimated_prop.boxplot('estimated_pi', by='true_pi', grid=False, return_type='axes')
    savefig('plot')
    close()

    figure()
    performance.boxplot(by='true_pi', grid=False)
    axhline(0.5, color='black')
    savefig('performance')
    close()

if __name__ == '__main__':
    with pushdir('gaussian-idealized-one-component'):
        plot_idealized_one_component()
    with pushdir('logistic-idealized-one-component'):
        plot_idealized_one_component('auprc')
