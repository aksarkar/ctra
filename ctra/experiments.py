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

    figure()
    results.boxplot(column=[k for k in results.columns if measure in k],
                    by='true_b', grid=False)
    if measure == 'score':
        axhline(0.5, color='black')
    savefig('performance')
    close()

def plot_one_component(measure):
    results = parse_results()
    plot_performance(results, measure)

    figure()
    results.boxplot(column='m0_b', by='true_b', grid=False,
                    return_type='axes')
    savefig('plot')
    close()

def plot_two_component(measure):
    results = parse_results()
    plot_performance(results, measure)

    true_log_odds = results.apply(lambda x: pandas.Series(scipy.special.logit(x['num_causal'] / x['simulation'].p)), 1)
    est_log_odds = (results['m0_b'] + results['m1_w']).apply(pandas.Series)
    est_log_odds['diff'] = est_log_odds[1] - est_log_odds[0]
    log_odds = true_log_odds.merge(right=est_log_odds, left_index=True, right_index=True)
    log_odds.columns = ['true_log_odds_0', 'true_log_odds_1', 'est_log_odds_0', 'est_log_odds_1', 'est_diff']
    equal_effects = log_odds[log_odds['true_log_odds_0'] != log_odds['true_log_odds_1']]
    equal_prop = log_odds[log_odds['true_log_odds_0'] == log_odds['true_log_odds_1']]

    if len(equal_prop) > 0:
        figure()
        equal_prop.boxplot(column='est_diff', by='true_log_odds_0', grid=False,
                           return_type='axes')
        savefig('equal-prop-diff')
        close()

        fig, ax = subplots(1, 2, sharey=True)
        equal_prop.boxplot(column='est_log_odds_0', by='true_log_odds_0',
                           ax=ax[0], grid=False, return_type='axes')
        equal_prop.boxplot(column='est_log_odds_1', by='true_log_odds_1',
                           ax=ax[1], grid=False, return_type='axes')
        savefig('equal-prop')
        close()

    figure()
    equal_effects.boxplot(column='est_diff', by='true_log_odds_0', grid=False,
                          return_type='axes')
    savefig('equal-effects-diff')
    close()

    fig, axes = subplots(1, 2, sharey=True)
    equal_effects.boxplot(column='est_log_odds_0', by='true_log_odds_0',
                          ax=axes[0], grid=False, return_type='axes')
    if numpy.isfinite(equal_effects['true_log_odds_1']).all():
        equal_effects.boxplot(column='est_log_odds_1', by='true_log_odds_1',
                              ax=axes[1], grid=False, return_type='axes')
    else:
        equal_effects.boxplot(column='est_log_odds_1', by='true_log_odds_0',
                              ax=axes[1], grid=False, return_type='axes')
    savefig('equal-effects')
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

def plot_real_annotations(measure):
    results = parse_results()
    plot_performance(results, measure)

    m1_w = results['m1_w'].apply(pandas.Series)
    annotations = list(m1_w.columns)
    m1_w['true_b'] = results['true_b']

    fig, ax = subplots(2, 1)
    fig.set_size_inches(30, 12);
    for a, (k, g) in zip(ax, m1_w.groupby('true_b')):
        a.set_title(k)
        a.axhline(y=0, color='black', linestyle='dashed')
        g.boxplot(column=annotations, grid=False, ax=a)
    xlabel('Annotation')
    ylabel('Log odds ratio')
    savefig('log-odds')
    close()

    m1_scale = 1 / (results['m0_c'] + results['m1_v']).apply(pandas.Series).apply(_softplus)

    fig = gcf();
    clf();
    fig.set_size_inches(11, 6);
    axhline(y=1, color='black', linestyle='dashed')
    m1_scale.T.boxplot(column=list(m1_scale.T.columns), grid=False)
    xlabel('Simulation trial')
    ylabel('Estimated effect size scale')
    savefig('scale-by-trial')
    close()

    fig = gcf();
    clf();
    fig.set_size_inches(30, 6);
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
    with pushdir('gaussian-realistic-one-component'):
        plot_one_component('score')
    with pushdir('gaussian-idealized-two-component'):
        plot_two_component('score')
    with pushdir('logistic-idealized-two-component'):
        plot_two_component('auprc')
    with pushdir('gaussian-realistic-synthetic-annotations'):
        plot_two_component('score')
    with pushdir('gaussian-realistic-Enh'):
        plot_real_annotations('score')
    with pushdir('gaussian-realistic-EnhClusters'):
        plot_real_annotations('score')
