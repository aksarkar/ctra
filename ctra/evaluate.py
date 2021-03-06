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
import pickle
import logging
import sys

import h5py
from matplotlib.pyplot import *
import numpy
import scipy.stats
import scipy.linalg
import sklearn.linear_model
import sklearn.metrics

import ctra

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
    req_args.add_argument('-m', '--model', choices=['gaussian', 'logistic', 'bslmm'], help='Type of model to fit', required=True)
    req_args.add_argument('-n', '--num-samples', type=int, help='Number of samples', required=True)
    req_args.add_argument('-p', '--num-variants', type=int, help='Number of genetic variants', required=True)
    req_args.add_argument('-v', '--pve', type=float, help='Total proportion of variance explained', required=True)
    req_args.add_argument('--validation', type=int, help='Hold out validation set size (for final fit)', required=True)

    input_args = parser.add_mutually_exclusive_group()
    input_args.add_argument('-G', '--load-oxstats', nargs='+', help='OXSTATS data sets (.sample, .gen.gz)', default=None)
    input_args.add_argument('-H', '--load-hdf5', help='HDF5 data set', default=None)

    annot_args = parser.add_argument_group()
    annot_args.add_argument('--annotation-matrix', help='Annotation matrix')
    annot_args.add_argument('--annotation-matrix-column', help='Column of annotation matrix to use for simulation', type=int, default=0)
    annot_args.add_argument('-A', '--annotation-vector', help='Annotation vector')

    output_args = parser.add_argument_group('Output', 'Writing out fitted models')
    output_args.add_argument('--write-result', help='Output file for pickled result', default=None)

    sim_args = parser.add_argument_group('Simulation', 'Parameters for generating synthetic data')
    sim_args.add_argument('--permute-causal', action='store_true', help='Permute causal indicator during generation (default: False)', default=False)
    sim_args.add_argument('--min-pve', type=float, help='Minimum PVE', default=1e-4)
    sim_args.add_argument('--min-maf', type=float, help='Minimum MAF', default=None)
    sim_args.add_argument('--max-maf', type=float, help='Maximum MAF', default=None)
    sim_args.add_argument('-K', '--prevalence', type=float, help='Binary phenotype case prevalence (assume Gaussian if omitted)', default=None)
    sim_args.add_argument('-P', '--study-prop', type=float, help='Binary phenotype case study proportion (assume 0.5 if omitted but prevalence given)', default=None)
    sim_args.add_argument('-s', '--seed', type=int, help='Random seed', default=0)

    parser.add_argument('--prior-mean-b', type=float, default=None)
    parser.add_argument('--prior-mean-c', type=float, default=None)
    parser.add_argument('--regularized', action='store_true', help='Fit regularized models', default=False)
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
    if args.model == 'logistic' and args.study_prop is None:
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
    if args.annotation_vector is not None:
        logger.debug('Loading annotation vector')
        a = numpy.loadtxt(args.annotation_vector).astype('int8')
        if a.shape[0] != args.num_variants:
            raise _A('{} variants present in annotations file, but {} specified'.format(a.shape[0], args.num_variants))
        s.load_annotations(a)
    elif args.annotation_matrix is not None:
        logger.debug('Loading annotation matrix')
        with gzip.open(args.annotation_matrix, 'rt') as f:
            a = numpy.loadtxt(f).astype('int8')
        if a.shape[0] != args.num_variants:
            raise _A('{} variants present in annotations file, but {} specified'.format(a.shape[0], args.num_variants))
        if args.annotation_matrix_column < 0:
            raise _A('Annotation column must be non-negative')
        if args.annotation_matrix_column > a.shape[1]:
            raise _A('{} columns present in annotation matrix, but column {} specified'.format(a.shape[1], args.annotation_matrix_column))
        s.load_annotations(a, args.annotation_matrix_column)
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
        if args.model == 'logistic':
            x, y = s.sample_case_control(n=args.num_samples, K=args.prevalence, P=args.study_prop)
        else:
            x, y = s.sample_gaussian(n=args.num_samples)
    # Hold out samples
    if args.model == 'logistic':
        # Randomly subsample hold out set
        validation = numpy.zeros(args.num_samples, dtype='bool')
        hold_out = s.random.choice(args.num_samples, args.validation, replace=False)
        validation[hold_out] = True
        x_validate = x[validation]
        y_validate = y[validation]
        # Permute the training set so minibatches are balanced in expectation
        perm = s.random.permutation(args.num_samples - args.validation)
        x = x[~validation][perm]
        y = y[~validation][perm]
    else:
        # Assume samples are exchangeable
        x_validate = x[-args.validation:]
        y_validate = y[-args.validation:]
        x = x[:-args.validation]
        y = y[:-args.validation]
    x -= x.mean(axis=0)
    x_validate -= x_validate.mean(axis=0)
    if args.model != 'logistic':
        y -= y.mean()
        y_validate -= y_validate.mean()
    return x, y, x_validate, y_validate

auprc = sklearn.metrics.average_precision_score

def _regularized(args, model, x, y, x_validate, y_validate, **kwargs):
    logger.info('Fitting regularized model {}'.format(model))
    m = model(**kwargs).fit(x, y)
    result = {'m': m,
              'training_set_score': m.score(x, y),
              'validation_set_score': m.score(x_validate, y_validate)}
    logger.info('Training score = {:.3f}'.format(result['training_set_score']))
    logger.info('Validation score = {:.3f}'.format(result['validation_set_score']))
    if args.model == 'logistic':
        result['training_set_auprc'] = auprc(y, m.predict_proba(x)[:,1])
        result['validation_set_auprc'] = auprc(y_validate, m.predict_proba(x_validate)[:,1])
        logger.info('Training set AUPRC = {:.3f}'.format(result['training_set_auprc']))
        logger.info('Validation set AUPRC = {:.3f}'.format(result['validation_set_auprc']))
    return result

def _fit(args, s, x, y, x_validate, y_validate):
    result = {'args': args,
              'simulation': s}

    if args.regularized:
        if args.model == 'logistic':
            result['logistic'] = _regularized(args, sklearn.linear_model.LogisticRegressionCV, x, y,
                                              x_validate, y_validate, penalty='l1', fit_intercept=True,
                                              solver='liblinear')
        else:
            result['lasso'] = _regularized(args, sklearn.linear_model.LassoCV, x, y, x_validate,
                                           y_validate)
            result['elastic_net'] = _regularized(args, sklearn.linear_model.ElasticNetCV, x, y, x_validate,
                                                 y_validate)

    model = {'gaussian': ctra.model.GaussianSGVB,
             'logistic': ctra.model.LogisticSGVB,
             'bslmm': ctra.model.VBSLMM
    }[args.model]

    def fit(params, m0=None):
        params = numpy.array(params)
        stoch_samples, log_learning_rate, max_epochs, rho = params.astype('float32')
        m = model(x, y, s.annot_matrix, m0=m0,
                  stoch_samples=int(stoch_samples),
                  learning_rate=numpy.exp(log_learning_rate),
                  rho=rho, random_state=s.random,
                  prior_mean_b=args.prior_mean_b,
                  prior_mean_c=args.prior_mean_c)
        # Multiply by 10 since we check ELBO, loss every 10 epochs
        m.fit(max_epochs=10 * int(max_epochs), xv=x_validate, yv=y_validate)
        return m

    opt = {'x_opt': [50, -2.5, 40, 0.9]}
    result.update(opt)

    logger.info('Fitting genome-wide model')
    m0 = fit(numpy.array(opt['x_opt']))
    result['m0_theta'] = m0.theta
    result['m0_pip'] = m0.pip
    result['m0_b'] = m0.b
    result['m0_c'] = m0.c
    result['m0_training_set_score'] = numpy.asscalar(m0.score(x, y))
    result['m0_validation_set_score'] = numpy.asscalar(m0.score(x_validate, y_validate))
    logger.info('Training set score = {:.3f}'.format(result['m0_training_set_score']))
    logger.info('Validation set score = {:.3f}'.format(result['m0_validation_set_score']))
    if args.model == 'logistic':
        result['m0_training_set_auprc'] = auprc(y, m0.predict_proba(x))
        result['m0_validation_set_auprc'] = auprc(y_validate, m0.predict_proba(x_validate))
        logger.info('Training set AUPRC = {:.3f}'.format(result['m0_training_set_auprc']))
        logger.info('Validation set AUPRC = {:.3f}'.format(result['m0_validation_set_auprc']))
    logger.info('Posterior mode genome-wide log odds = {}'.format(m0.b))

    if len(args.annotation) > 1:
        logger.info('Fitting annotation model')
        m1 = fit(numpy.array(opt['x_opt']), m0=m0)
        result['m1_theta'] = m1.theta
        result['m1_pip'] = m1.pip
        result['m1_w'] = m1.w
        result['m1_v'] = m1.v
        result['m1_training_set_score'] = numpy.asscalar(m1.score(x, y))
        result['m1_validation_set_score'] = numpy.asscalar(m1.score(x_validate, y_validate))
        logger.info('Training set score = {:.3f}'.format(result['m1_training_set_score']))
        logger.info('Validation set score = {:.3f}'.format(result['m1_validation_set_score']))
        if args.model == 'logistic':
            result['m1_training_auprc'] = auprc(y, m1.predict_proba(x))
            result['m1_validation_auprc'] = auprc(y_validate, m1.predict_proba(x_validate))
            logger.info('Training set AUPRC = {:.3f}'.format(result['m1_training_auprc']))
            logger.info('Validation set AUPRC = {:.3f}'.format(result['m1_validation_auprc']))
        logger.info('Posterior mode annotation log odds = {}'.format(m1.w))

    if args.write_result is not None:
        with open(args.write_result, 'wb') as f:
            pickle.dump(result, f)

    if args.interact:
        code.interact(banner='', local=dict(globals(), **locals()))

def evaluate():
    """Entry point for simulations on synthetic genotypes/phenotypes/annotations"""
    args = _parser().parse_args()
    _validate(args)
    logging.getLogger('ctra').setLevel(args.log_level)
    with ctra.simulation.simulation(args.num_variants, args.pve, args.annotation, args.seed) as s:
        args.num_samples += args.validation
        x, y, x_validate, y_validate = _load_data(args, s)
        _fit(args, s, x, y, x_validate, y_validate)
