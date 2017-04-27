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
    req_args.add_argument('-m', '--model', choices=['gaussian', 'probit', 'logistic'], help='Type of model to fit', required=True)
    req_args.add_argument('-n', '--num-samples', type=int, help='Number of samples', required=True)
    req_args.add_argument('-p', '--num-variants', type=int, help='Number of genetic variants', required=True)
    req_args.add_argument('--validation', type=int, help='Hold out validation set for posterior predictive check', required=True)
    req_args.add_argument('-v', '--pve', type=float, help='Total proportion of variance explained', required=True)

    input_args = parser.add_mutually_exclusive_group()
    input_args.add_argument('-G', '--load-oxstats', nargs='+', help='OXSTATS data sets (.sample, .gen.gz)', default=None)
    input_args.add_argument('-H', '--load-hdf5', help='HDF5 data set', default=None)

    data_args = parser.add_argument_group('Data', 'Data processing options')
    data_args.add_argument('-A', '--load-annotations', help='Annotation vector')
    data_args.add_argument('--center', action='store_true', help='Center covariates to have zero mean', default=False)

    output_args = parser.add_argument_group('Output', 'Writing out fitted models')
    output_args.add_argument('--diagnostic', action='store_true')
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

    vb_args = parser.add_argument_group('Variational Bayes', 'Parameters for tuning Variational Bayes optimization')
    vb_args.add_argument('-r', '--learning-rate', type=float, help='Learning rate for SGD', default=0.1)
    vb_args.add_argument('-b', '--minibatch-size', type=int, help='Minibatch size for SGD', default=100)
    vb_args.add_argument('-i', '--max-epochs', type=int, help='Polling interval for SGD', default=20)
    vb_args.add_argument('-t', '--trace', action='store_true', help='Store trace')

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
    if args.prevalence is None and args.model in ('probit', 'logistic'):
        raise _A('Prevalence must be specified for logistic model')
    if args.min_maf is not None and (numpy.isclose(args.min_maf, 0) or args.min_maf < 0):
        raise _A('Minimum MAF must be larger than float tolerance')
    if args.max_maf is not None and args.max_maf > .5:
        raise _A('Maximum MAF must be less than or equal to 0.5')
    if args.learning_rate <= 0:
        raise _A('Learning rate must be positive')
    if args.minibatch_size <= 0:
        raise _A('Minibatch size must be positive')
    if args.minibatch_size > args.num_samples:
        logger.warn('Setting minibatch size to sample size')
        args.minibatch_size = args.num_samples
    if args.max_epochs <= 0:
        raise _A('Maximum epochs must be positive')
    max_ = args.num_variants // len(args.annotation)
    for i, (num, var) in enumerate(args.annotation):
        if not 0 <= num <= max_:
            raise _A('Annotation {} must have between 0 and {} causal variants ({} specified)'.format(i, max_, num))
        if var <= 0:
            raise _A('Annotation {} must have positive effect size variance'.format(i))

def _load_data(args, s):
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
    if args.center:
        x -= x.mean(axis=0)
        y -= y.mean()
    return x, y

def _fit(args, s, x, y, x_validate=None, y_validate=None):
    model = {'gaussian': ctra.model.GaussianSGVB,
             'logistic': ctra.model.LogisticSGVB,
             'probit': ctra.model.ProbitSGVB,}[args.model]
    m = model(x, y, s.annot, learning_rate=args.learning_rate,
              random_state=s.random,
              minibatch_n=args.minibatch_size)
    def loss(loc):
        return m.fit(loc=loc, max_epochs=args.max_epochs, xv=x_validate, yv=y_validate).validation_loss
    opt = robo.fmin.bayesian_optimization(loss, numpy.array([-7]), numpy.array([0]), num_iterations=40)
    m.fit(loc=opt['x_opt'], max_epochs=args.max_epochs, xv=x_validate, yv=y_validate)
    logger.info('Training set correlation = {:.3f}'.format(numpy.asscalar(m.score(x, y))))
    logger.info('Validation set correlation = {:.3f}'.format(numpy.asscalar(m.score(x_validate, y_validate))))
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
    logger.info('Writing posterior mean pi')
    numpy.savetxt(sys.stdout.buffer, m.pi, fmt='%.3g')

def evaluate():
    """Entry point for simulations on synthetic genotypes/phenotypes/annotations"""
    args = _parser().parse_args()
    _validate(args)
    logging.getLogger('ctra').setLevel(args.log_level)
    with ctra.simulation.simulation(args.num_variants, args.pve, args.annotation, args.seed) as s:
        if args.validation is not None:
            args.num_samples += args.validation
        x, y = _load_data(args, s)
        if args.validation is not None:
            if args.model == 'gaussian':
                x_validate = x[-args.validation:]
                y_validate = y[-args.validation:]
                x = x[:-args.validation]
                y = y[:-args.validation]
            else:
                validation = numpy.zeros(args.num_samples, dtype='bool')
                validation[s.random.choice(args.num_samples, args.validation, replace=False)] = True
                x_validate = x[validation]
                y_validate = y[validation]
                x = x[~validation]
                y = y[~validation]
            if args.center:
                x_validate -= x_validate.mean(axis=0)
                y_validate -= y_validate.mean()
                x -= x.mean(axis=0)
                y -= y.mean()
        else:
            x_validate = None
            y_validate = None
        _fit(args, s, x, y, x_validate, y_validate)
