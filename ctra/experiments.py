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
    results['num_causal'] = [numpy.array([x[0] for x in a.annotation]) for a in results['args']]
    results['true_b'] = [scipy.special.logit(r['num_causal'].sum() / r['args'].num_variants)
                         for _, r in results.iterrows()]
    return results

def plot_performance(results, measure):
    if measure not in ('score', 'auprc'):
        raise ArgumentError
    label = {'score': 'Coefficient of determination',
             'auprc': 'Area under precision-recall curve'}

    columns = [k for k in results.columns if measure in k]
    fig = gcf()
    clf()
    ax = results.boxplot(column=columns, by='true_b', grid=False, return_type='axes',
                         figsize=(3.5 * len(columns), 3), layout=(1, len(columns)))
    for a in ax:
        # Remove pandas nonsense
        a.get_figure().texts = []
        title = ' '.join(a.get_title().split('_')[1:3]).capitalize()
        model = '$m_{}$'.format(a.get_title().split('_')[0][-1])
        a.set_title('{} {}'.format(model, title))
        labels = ['{:.3f}'.format(float(x.get_text())) for x in ax[0].get_xaxis().get_ticklabels()]
        a.get_xaxis().set_ticklabels(labels)
        a.set_xlabel('Causal log odds')
        a.set_ylabel(label[measure])
        if measure == 'score':
            a.axhline(0.5, color='black')
    savefig('performance')
    close()

def plot_one_component(measure):
    results = parse_results()
    plot_performance(results, measure)

    gcf()
    clf()
    ax = results.boxplot(column='m0_b', by='true_b', grid=False,
                         return_type='axes', figsize=(3, 3))
    a = ax[0]
    a.get_figure().texts = []
    labels = ['{:.3f}'.format(float(x.get_text())) for x in a.get_xaxis().get_ticklabels()]
    a.set_title('')
    a.get_xaxis().set_ticklabels(labels)
    a.set_xlabel('True log odds')
    a.set_ylabel('Estimated log odds')
    gcf().subplots_adjust(left=0.25)
    savefig('plot')
    close()

def plot_two_component(measure):
    results = parse_results()
    plot_performance(results, measure)

    equal_prop = results['num_causal'].apply(lambda x: x[0] == x[1])
    fig, ax = subplots(2, 4)
    fig.set_size_inches(12, 6)
    for row, facet in zip(ax, [~equal_prop, equal_prop]):
        for i, (k, g) in enumerate(results[facet].groupby('true_b')):
            row[2 * i].boxplot(numpy.array(g['m1_w'].apply(pandas.Series)))
            row[2 * i].axhline(y=0, color='black')
            expected_log_odds_ratio = (k - g['m0_b']).mean()
            row[2 * i].axhline(y=expected_log_odds_ratio, color='red')
            row[2 * i].set_ylabel('Log odds ratio')
            row[2 * i].set_xlabel('Annotation')
            row[2 * i].set_xticklabels([0, 1])
            row[2 * i].set_title('Causal log odds={:.3f}'.format(k))

            row[2 * i + 1].boxplot(numpy.array(g['m1_v'].apply(pandas.Series)))
            row[2 * i + 1].axhline(y=0, color='black')
            row[2 * i + 1].set_ylabel('Logit change in precision')
            row[2 * i + 1].set_xlabel('Annotation')
            row[2 * i + 1].set_xticklabels([0, 1])
            row[2 * i + 1].set_title('Causal log odds={:.3f}'.format(k))
    fig.subplots_adjust(wspace=.5, hspace=.5)
    savefig('two-component')
    close()

def _scalar_softplus(x):
    """Numerically stable implementation of log(1 + exp(x))

    This is the link function for the scale parameter

    """
    if x < -30.0:
        return 0.0
    elif x > 30.0:
        return x
    else:
        return numpy.log1p(numpy.exp(x))

_softplus = numpy.vectorize(_scalar_softplus)

def plot_synthetic_annotations():
    results = parse_results()
    plot_performance(results, 'score')

    m1_w = results['m1_w'].apply(pandas.Series)
    annotations = list(m1_w.columns)
    m1_w['m0_b'] = results['m0_b']
    m1_w['true_b'] = results['true_b']
    m1_w['annotation'] = [r['args'].annotation_vector for _, r in results.iterrows()]

    fig, ax = subplots(3, 2)
    fig.set_size_inches(6, 9)
    keys = []
    for a, (k, g) in zip(ax.flatten(), m1_w.groupby(['true_b', 'annotation'])):
        keys.append(k)
        a.boxplot(numpy.array(g[annotations]))
        a.axhline(y=0, color='black')
        expected_log_odds_ratio = (k[0] - g['m0_b']).mean()
        a.axhline(y=expected_log_odds_ratio, color='red')
        a.set_xticklabels([0, 1])
        null_log_odds_ratio = (-numpy.log(results.loc[0, 'simulation'].p) - g['m0_b']).mean()
        a.axhline(y=null_log_odds_ratio, color='red', linestyle='dashed')
    keys = numpy.array(keys).reshape(3, 2, 2)
    for row, logodds in zip(ax, keys[:,0,0]):
        row[0].set_ylabel('Log odds ratio')
        row[1].set_ylabel('Causal log odds={:.3f}'.format(float(logodds)))
        row[1].yaxis.set_label_position('right')
    for col, annotation in zip(ax.T, keys[0,:,1]):
        size = annotation.split('-')[-1][:-4].upper()
        col[0].set_xlabel('Element size={}'.format(size))
        col[0].xaxis.set_label_position('top')
        col[-1].set_xlabel('Annotation')
    savefig('log-odds')
    close()

def plot_real_annotations():
    results = parse_results()
    plot_performance(results, 'score')

    m1_w = results['m1_w'].apply(pandas.Series)
    annotations = list(m1_w.columns)
    m1_w['m0_b'] = results['m0_b']
    m1_w['true_b'] = results['true_b']
    m1_w['annotation_matrix_column'] = results['args'].apply(lambda x: x.annotation_matrix_column)

    groups = m1_w.groupby(['annotation_matrix_column', 'true_b'])
    fig, ax = subplots(len(groups), 1)
    fig.set_size_inches(11.5, 2 * len(groups))
    for a, (k, g) in zip(ax, groups):
        a.set_title('Causal annotation = {}, genome-wide causal log odds = {:.3f}'.format(*k))
        a.set_ylabel('Log odds ratio')
        g.boxplot(column=annotations, grid=False, ax=a)
        a.axhline(y=0, color='black')
        expected_log_odds_ratio = (k[1] - g['m0_b']).mean()
        a.axhline(y=expected_log_odds_ratio, color='red')
        a.set_xticklabels([])
    xlabel('Annotation')
    savefig('log-odds')
    close()

    m1_scale = 1 / (results['m0_c'] + results['m1_v']).apply(pandas.Series).apply(_softplus)

    fig = gcf()
    clf()
    fig.set_size_inches(11, 6)
    axhline(y=1, color='black', linestyle='dashed')
    m1_scale.T.boxplot(column=list(m1_scale.T.columns), grid=False)
    xlabel('Simulation trial')
    ylabel('Estimated effect size scale')
    savefig('scale-by-trial')
    close()

    fig = gcf()
    clf()
    fig.set_size_inches(30, 6)
    axhline(y=1, color='black', linestyle='dashed')
    m1_scale.boxplot(column=list(m1_scale.columns), grid=False)
    xlabel('Annotation')
    ylabel('Estimated effect size scale')
    savefig('scale-by-annotation')
    close()

if __name__ == '__main__':
    with pushdir('gaussian-idealized-one-component'):
        plot_one_component('score')
    with pushdir('logistic-idealized-one-component'):
        plot_one_component('auprc')

    with pushdir('gaussian-idealized-two-component'):
        plot_two_component('score')
    with pushdir('logistic-idealized-two-component'):
        plot_two_component('auprc')

    with pushdir('gaussian-realistic-one-component'):
        plot_one_component('score')
    with pushdir('gaussian-realistic-two-component'):
        plot_two_component('score')

    with pushdir('gaussian-realistic-synthetic-annotations'):
        plot_synthetic_annotations()

    with pushdir('gaussian-realistic-Enh'):
        plot_real_annotations()
    with pushdir('gaussian-realistic-EnhClusters'):
        plot_real_annotations()
