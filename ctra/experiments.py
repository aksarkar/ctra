import contextlib
import glob
import pickle
import os
import os.path

from matplotlib.pyplot import *
import numpy
import scipy.special

switch_backend('pdf')

@contextlib.contextmanager
def pushdir(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    yield
    os.chdir(cwd)

def plot_idealized_one_component(performance='score'):
    if performance not in ('score', 'auprc'):
        raise ArgumentError
    files = glob.glob(os.path.join('[0-9]*.pkl'))
    estimated_prop = []
    performance = []
    for f in files:
        with open(f, 'rb') as f:
            result = pickle.load(f)
            num_causal = len(numpy.where(result['simulation'].theta != 0)[0])
            num_variants = result['simulation'].p
            estimated_prop.append([scipy.special.logit(num_causal / num_variants), result['m0_b']])
            score.append([scipy.special.logit(num_causal / num_variants),
                          result['m0_training_set_{}'].format(performance),
                          result['m0_validation_set_{}'].format(performance),
                          ])
    estimated_prop = numpy.array(estimated_prop)
    figure()
    boxplot(estimated_prop)
    axhline(scipy.special.logit(.01), color='black')
    axhline(scipy.special.logit(.1), color='black')
    savefig('plot')
    close()
    figure()
    boxplot(performance)
    axhline(0.5, color='black')
    savefit('plot')
    savefig('performance')
    close()

if __name__ == '__main__':
    with pushdir('gaussian-idealized-one-component'):
        plot_idealized_one_component()
    with pushdir('logistic-idealized-one-component'):
        plot_idealized_one_component()
