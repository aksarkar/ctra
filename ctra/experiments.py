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
