"""Evaluate the accuracy of the model

This module provides the entry point for the simulation study. This is needed
because the development compute environment has strict memory limits and
low-level concurrency primitives.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import argparse
import code
import contextlib
import gzip
import itertools
import logging
import sys

import h5py
from matplotlib.pyplot import *
import numpy
import robo.fmin
import scipy.stats
import scipy.linalg
import sklearn.linear_model
import sklearn.metrics

import ctra.algorithms
import ctra.formats
import ctra.model
import ctra.simulation

logger = logging.getLogger(__name__)
_A = argparse.ArgumentTypeError

# Set up things for plotting interactively
switch_backend('pdf')

def _parser():
    def Annotation(arg):
        num, var = arg.split(',')
        return (int(num), numpy.sqrt(float(var)))
    parser = argparse.ArgumentParser(description='Evaluate the model on synthetic data')
    req_args = parser.add_argument_group('Required arguments', '')
    req_args.add_argument('-a', '--annotation', type=Annotation, action='append', help="""Annotation parameters (num. causal, effect size var.) separated by ','. Repeat for additional annotations.""", default=[], required=True)
    req_args.add_argument('-m', '--model', choices=['gaussian', 'logistic'], help='Type of model to fit', required=True)
    req_args.add_argument('-n', '--num-samples', type=int, help='Number of samples', required=True)
    req_args.add_argument('-p', '--num-variants', type=int, help='Number of genetic variants', required=True)
    req_args.add_argument('-v', '--pve', type=float, help='Total proportion of variance explained', required=True)
    req_args.add_argument('--test', type=int, help='Test set size (for Bayesian optimization)', required=True)
    req_args.add_argument('--validation', type=int, help='Hold out validation set size (for final fit)', required=True)

    input_args = parser.add_mutually_exclusive_group()
    input_args.add_argument('-G', '--load-oxstats', nargs='+', help='OXSTATS data sets (.sample, .gen.gz)', default=None)
    input_args.add_argument('-H', '--load-hdf5', help='HDF5 data set', default=None)
    input_args.add_argument('-A', '--load-annotations', help='Annotation vector')

    output_args = parser.add_argument_group('Output', 'Writing out fitted models')
    output_args.add_argument('--write-model', help='Prefix for pickled model', default=None)
    output_args.add_argument('--plot', help='File to plot active samples to', default=None)

    sim_args = parser.add_argument_group('Simulation', 'Parameters for generating synthetic data')
    sim_args.add_argument('--permute-causal', action='store_true', help='Permute causal indicator during generation (default: False)', default=False)
    sim_args.add_argument('--min-pve', type=float, help='Minimum PVE', default=1e-4)
    sim_args.add_argument('--min-maf', type=float, help='Minimum MAF', default=None)
    sim_args.add_argument('--max-maf', type=float, help='Maximum MAF', default=None)
    sim_args.add_argument('-K', '--prevalence', type=float, help='Binary phenotype case prevalence (assume Gaussian if omitted)', default=None)
    sim_args.add_argument('-P', '--study-prop', type=float, help='Binary phenotype case study proportion (assume 0.5 if omitted but prevalence given)', default=None)
    sim_args.add_argument('-s', '--seed', type=int, help='Random seed', default=0)

    parser.add_argument('-j', '--jacknife', type=int, help='Number of jacknifes', default=0)
    parser.add_argument('-l', '--log-level', choices=['INFO', 'DEBUG'], help='Log level', default='INFO')
    parser.add_argument('--interact', action='store_true', help='Drop into interactive shell after fitting the model', default=False)

    return parser

def _validate(args):
    """Validate command line arguments"""
    if args.num_samples <= 0:
        raise _A('Number of samples must be positive')
    if args.num_variants <= 0:
        raise _A('Number of variants must be positive')
    if not 0 <= args.pve <= 1:
        raise _A('PVE must be in [0, 1]')
    if args.prevalence is None and args.study_prop is not None:
        raise _A('Case prevalence must be specified if case study proportion is specified')
    if args.prevalence is not None and not 0 <= args.prevalence <= 1:
        raise _A('Case prevalence must be in [0, 1]')
    if args.study_prop is not None and not 0 <= args.study_prop <= 1:
        raise _A('Case study proportion must be in [0, 1]')
    if args.model != 'gaussian' and args.study_prop is None:
        args.study_prop = 0.5
        logger.warn('Assuming study prevalence 0.5')
    if args.prevalence is None and args.model == 'logistic':
        raise _A('Prevalence must be specified for model "{}"'.format(args.model))
    if args.min_maf is not None and (numpy.isclose(args.min_maf, 0) or args.min_maf < 0):
        raise _A('Minimum MAF must be larger than float tolerance')
    if args.max_maf is not None and args.max_maf > .5:
        raise _A('Maximum MAF must be less than or equal to 0.5')
    max_ = args.num_variants // len(args.annotation)
    for i, (num, var) in enumerate(args.annotation):
        if not 0 <= num <= max_:
            raise _A('Annotation {} must have between 0 and {} causal variants ({} specified)'.format(i, max_, num))
        if var <= 0:
            raise _A('Annotation {} must have positive effect size variance'.format(i))

def _load_data(args, s):
    # Prepare the simulation parameters
    if args.min_maf is not None or args.max_maf is not None:
        if args.min_maf is None:
            args.min_maf = 0.01
        if args.max_maf is None:
            args.max_maf = 0.05
        s.sample_mafs(args.min_maf, args.max_maf)
    if args.load_annotations is not None:
        logger.debug('Loading pre-computed annotations')
        a = numpy.loadtxt(args.load_annotations).astype('int')
        if a.shape[0] != args.num_variants:
            raise _A('{} variants present in annotations file, but {} specified'.format(a.shape[0], args.num_variants))
        s.load_annotations(a)
    if args.permute_causal:
        logger.debug('Generating effects with permuted causal indicator')
        s.sample_effects(pve=args.pve, annotation_params=args.annotation, permute=True)
    # Load/generate the genotypes and phenotypes
    if args.load_oxstats:
        logger.debug('Loading OXSTATS datasets')
        with contextlib.ExitStack() as stack:
            data = [stack.enter_context(ctra.formats.oxstats_genotypes(*a))
                    for a in ctra.algorithms.kwise(args.load_oxstats, 2)]
            samples = list(itertools.chain.from_iterable(s for _, _, s, _ in data))
            merged = ctra.formats.merge_oxstats([d for _, _, _, d in data])
            probs = ([float(x) for x in row[5:]] for row in merged)
            if args.num_samples > len(samples):
                logger.error('{} individuals present in OXSTATS data, but {} were specified'.format(len(samples), args.num_samples))
                sys.exit(1)
            x = numpy.array(list(itertools.islice(probs, args.num_variants)))
        p, n = x.shape
        if p < args.num_variants:
            logger.error('{} variants present in OXSTATS data, but {} were specified'.format(p, args.num_variants))
            sys.exit(1)
        n = n // 3
        x = (x.reshape(p, -1, 3) * numpy.array([0, 1, 2])).sum(axis=2).T[:n,:]
        s.estimate_mafs(x)
        y = s.compute_liabilities(x)
    elif args.load_hdf5:
        with h5py.File(args.load_hdf5) as f:
            logger.debug('Loading HDF5 dataset')
            x = f['dosage'][:args.num_samples, :args.num_variants]
            logger.debug('Re-computing liabilities')
            s.estimate_mafs(x)
            y = s.compute_liabilities(x)
    else:
        if args.model == 'gaussian':
            x, y = s.sample_gaussian(n=args.num_samples)
        else:
            x, y = s.sample_case_control(n=args.num_samples, K=args.prevalence, P=args.study_prop)
    # Hold out samples
    hold_out_n = args.test + args.validation
    if args.model == 'gaussian':
        # Assume samples are exchangeable
        x_test = x[-hold_out_n:-args.validation]
        y_test = y[-hold_out_n:-args.validation]
        x_validate = x[-args.validation:]
        y_validate = y[-args.validation:]
        x = x[:-hold_out_n]
        y = y[:-hold_out_n]
    else:
        # Randomly subsample hold out set
        hold_out = s.random.choice(args.num_samples, hold_out_n, replace=False)
        test = numpy.zeros(args.num_samples, dtype='bool')
        test[hold_out[:args.test]] = True
        validation = numpy.zeros(args.num_samples, dtype='bool')
        validation[hold_out[args.test:]] = True
        x_test = x[test]
        y_test = y[test]
        x_validate = x[validation]
        y_validate = y[validation]
        x = x[~hold_out]
        y = y[~hold_out]
    x -= x.mean(axis=0)
    x_test -= x_test.mean(axis=0)
    x_validate -= x_validate.mean(axis=0)
    if args.model == 'gaussian':
        y -= y.mean()
        y_test -= y_test.mean()
        y_validate -= y_validate.mean()
    return x, y, x_test, y_test, x_validate, y_validate

auprc = sklearn.metrics.average_precision_score

def _regularized(args, model, x, y, x_validate, y_validate, **kwargs):
    logger.info('Fitting regularized model {}'.format(model))
    m = model(**kwargs).fit(x, y)
    logger.info('Training score = {:.3f}'.format(m.score(x, y)))
    logger.info('Validation score = {:.3f}'.format(m.score(x_validate, y_validate)))
    if args.model != 'gaussian':
        logger.info('Training set AUPRC = {:.3f}'.format(auprc(y, m.predict_proba(x)[:,1])))
        logger.info('Validation set AUPRC = {:.3f}'.format(auprc(y_validate, m.predict_proba(x_validate)[:,1])))

def _fit(args, s, x, y, x_test, y_test, x_validate, y_validate):
    if args.model == 'gaussian':
        _regularized(args, sklearn.linear_model.Lasso, x, y, x_validate,
                     y_validate)
        _regularized(args, sklearn.linear_model.ElasticNet, x, y, x_validate,
                     y_validate)
    else:
        _regularized(args, sklearn.linear_model.LogisticRegression, x, y,
                     x_validate, y_validate, penalty='l1', fit_intercept=True,
                     solver='liblinear')

    logger.info('Performing Bayesian optimization')
    model = {'gaussian': ctra.model.GaussianSGVB,
             'logistic': ctra.model.LogisticSGVB,
    }[args.model]

    def fit(params, drop=None, b=None):
        stoch_samples, learning_rate, minibatch_size, max_epochs, rho = params.astype('float32')
        if drop is not None:
            n = y.shape[0]
            weights = numpy.ones(y.shape)
            weights[drop] = 0
        else:
            weights = None
        m = model(x, y, s.annot, b=b, weights=weights, stoch_samples=int(stoch_samples),
                  learning_rate=learning_rate, minibatch_n=int(minibatch_size),
                  rho=rho, random_state=s.random)
        # Multiply by 10 since we check ELBO, loss every 10 epochs
        m.fit(max_epochs=10 * int(max_epochs), xv=x_validate, yv=y_validate)
        return m

    def loss(params):
        m = fit(params)
        return m.validation_loss

    # Find the optimal learning parameters (minimum test loss)
    # stoch_samples, learning_rate, minibatch_size, max_epochs, rho
    lower_bound = numpy.array([1, 0.01, 50, 2, 0.5])
    upper_bound = numpy.array([50, 1, 200, 20, 0.9])
    logger.info('Performing Bayesian optimization')
    opt = robo.fmin.bayesian_optimization(loss, lower_bound, upper_bound, num_iterations=40)
    logger.info('Optimal learning parameters = {}'.format(opt))

    logger.info('Fitting genome-wide model')
    m0 = fit(numpy.array(opt['x_opt']))
    logger.info('Training set score = {:.3f}'.format(numpy.asscalar(m0.score(x, y))))
    logger.info('Validation set score = {:.3f}'.format(numpy.asscalar(m0.score(x_validate, y_validate))))
    if args.model != 'gaussian':
        logger.info('Training set AUPRC = {:.3f}'.format(auprc(y, m0.predict_proba(x))))
        logger.info('Validation set AUPRC = {:.3f}'.format(auprc(y_validate, m0.predict_proba(x_validate))))
    logger.info('Posterior mode pi = {}'.format(scipy.special.expit(m0.b)))

    if len(args.annotation) > 1:
        logger.info('Fitting annotation model')
        m1 = fit(numpy.array(opt['x_opt']), b=m0.b)
        logger.info('Training set score = {:.3f}'.format(numpy.asscalar(m1.score(x, y))))
        logger.info('Validation set score = {:.3f}'.format(numpy.asscalar(m1.score(x_validate, y_validate))))
        if args.model != 'gaussian':
            logger.info('Training set AUPRC = {:.3f}'.format(auprc(y, m1.predict_proba(x))))
            logger.info('Validation set AUPRC = {:.3f}'.format(auprc(y_validate, m1.predict_proba(x_validate))))
        logger.info('Posterior mode annotation log odds ratio = {}'.format(m1.w))

    if args.jacknife > 0:
        logger.info('Starting jacknife estimates')
        jacknife_se = numpy.array([fit(numpy.array(opt['x_opt']), drop=i).pi for i in s.random.choice(y.shape[0], 100)]).std()
        logger.info('Jacknife SE = {}'.format(jacknife_se))

    if args.write_model is not None:
        with open('{}'.format(args.write_model), 'w') as f:
            for row in zip(m.pip, m.theta, m.theta_var, s.maf, s.theta):
                print('\t'.join('{:.3g}'.format(numpy.asscalar(x)) for x in row), file=f)
    if args.plot is not None:
        q = numpy.logical_or(m.pip > 0.1, s.theta != 0)
        nq = numpy.count_nonzero(q)
        fig, ax = subplots(4, 1)
        fig.set_size_inches(6, 8)
        xlabel('True and false positive variants')
        ax[0].bar(range(nq), s.maf[q])
        ax[0].set_ylabel('MAF')
        ax[1].bar(range(nq), s.theta[q])
        ax[1].set_ylabel('True effect size')
        ax[2].bar(range(nq), m.theta[q])
        ax[2].set_ylabel('Estimated effect size')
        ax[3].bar(range(nq), m.pip[q])
        ax[3].set_ylabel('PIP')
        savefig('{}-pip.pdf'.format(args.plot))
        close()
    if args.interact:
        code.interact(banner='', local=dict(globals(), **locals()))

def evaluate():
    """Entry point for simulations on synthetic genotypes/phenotypes/annotations"""
    args = _parser().parse_args()
    _validate(args)
    logging.getLogger('ctra').setLevel(args.log_level)
    with ctra.simulation.simulation(args.num_variants, args.pve, args.annotation, args.seed) as s:
        args.num_samples += args.test
        args.num_samples += args.validation
        x, y, x_test, y_test, x_validate, y_validate = _load_data(args, s)
        _fit(args, s, x, y, x_test, y_test, x_validate, y_validate)
