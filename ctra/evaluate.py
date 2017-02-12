"""Evaluate the accuracy of the model

This module provides the entry point for the simulation study. This is needed
because the development compute environment has strict memory limits and
low-level concurrency primitives.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import argparse
import code
import collections
import contextlib
import gzip
import itertools
import logging
import pdb
import pprint
import os
import os.path
import signal
import sys

from matplotlib.pyplot import *
import numpy
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

def interrupt(s, f):
    """Catch SIGINT to terminate model fitting early"""
    signal.signal(signal.SIGINT, old_sigint)
    code.interact(banner='', local=dict(globals(), **locals()))

old_sigint = signal.signal(signal.SIGINT, interrupt)

def _parser():
    def Annotation(arg):
        num, var = arg.split(',')
        return (int(num), numpy.sqrt(float(var)))
    parser = argparse.ArgumentParser(description='Evaluate the model on synthetic data')
    req_args = parser.add_argument_group('Required arguments', '')
    req_args.add_argument('-a', '--annotation', type=Annotation, action='append', help="""Annotation parameters (num. causal, effect size var.) separated by ','. Repeat for additional annotations.""", default=[], required=True)
    req_args.add_argument('-m', '--model', choices=['gaussian', 'logistic'], help='Type of model to fit', required=True)
    req_args.add_argument('-M', '--method', choices=['pcgc', 'coord', 'varbvs', 'dsvi', 'sklearn'], help='Method to fit model', required=True)
    req_args.add_argument('-O', '--outer-method', choices=['is', 'wsabi'], help='Outer method (for hyperparameters)', required=True, default='is')
    req_args.add_argument('-n', '--num-samples', type=int, help='Number of samples', required=True)
    req_args.add_argument('-p', '--num-variants', type=int, help='Number of genetic variants', required=True)
    req_args.add_argument('-v', '--pve', type=float, help='Total proportion of variance explained', required=True)

    data_args = parser.add_argument_group('Input/output', 'Loading and writing data files')
    data_args.add_argument('--load-data', help='Directory to load data', default=None)
    data_args.add_argument('-G', '--load-oxstats', nargs='+', help='OXSTATS data sets (.sample, .gen.gz)', default=None)
    data_args.add_argument('-A', '--load-annotations', help='Annotation vector')
    data_args.add_argument('--write-data', help='Directory to write out data', default=None)
    data_args.add_argument('--write-sfba-data', help='Directory to write out data for SFBA', default=None)
    data_args.add_argument('--write-weights', help='Directory to write out importance weights', default=None)
    data_args.add_argument('--plot', help='File to plot active samples to', default=None)
    data_args.add_argument('--fit-null', action='store_true', help='Fit null model', default=False)
    data_args.add_argument('--bayes-factor', action='store_true', help='Compute Bayes factor', default=False)
    data_args.add_argument('--center', action='store_true', help='Center covariates to have zero mean', default=False)
    data_args.add_argument('--normalize', action='store_true', help='Center and scale covariates to have zero mean and variance one', default=False)
    data_args.add_argument('--rotate', action='store_true', help='Rotate data to orthogonalize covariates', default=False)
    data_args.add_argument('-l', '--log-level', choices=['INFO', 'DEBUG'], help='Log level', default='INFO')
    data_args.add_argument('--interact', action='store_true', help='Drop into interactive shell after fitting the model', default=False)

    hyper_args = parser.add_mutually_exclusive_group()
    hyper_args.add_argument('--no-pool', action='store_false', dest='pool', help='Propose per-annotation pi and tau', default=True)
    hyper_args.add_argument('--propose-tau', action='store_true', help='Propose per-annotation tau with shared pi', default=False)

    wsabi_args = parser.add_argument_group('WSABI', 'Parameters for Warped Sequential Approximate Bayesian Inference')
    wsabi_args.add_argument('--init-samples', type=int, help='Inital random samples', default=10)
    wsabi_args.add_argument('--max-samples', type=int, help='Inital random samples', default=40)
    wsabi_args.add_argument('--wsabi-tolerance', type=float, help='Maximum posterior variance in model evidence (for active sampling)', default=1e-2)


    sim_args = parser.add_argument_group('Simulation', 'Parameters for generating synthetic data')
    sim_args.add_argument('--permute-causal', action='store_true', help='Permute causal indicator during generation (default: False)', default=False)
    sim_args.add_argument('--min-pve', type=float, help='Minimum PVE', default=1e-4)
    sim_args.add_argument('--min-maf', type=float, help='Minimum MAF', default=None)
    sim_args.add_argument('--max-maf', type=float, help='Maximum MAF', default=None)
    sim_args.add_argument('-K', '--prevalence', type=float, help='Binary phenotype case prevalence (assume Gaussian if omitted)', default=None)
    sim_args.add_argument('-P', '--study-prop', type=float, help='Binary phenotype case study proportion (assume 0.5 if omitted but prevalence given)', default=None)
    sim_args.add_argument('-s', '--seed', type=int, help='Random seed', default=0)

    vb_args = parser.add_argument_group('Variational Bayes', 'Parameters for tuning Variational Bayes optimization')
    vb_args.add_argument('--true-causal', action='store_true', help='Fix causal indicator to its true value (default: False)', default=False)
    vb_args.add_argument('--true-pve', action='store_true', help='Fix hyperparameter PVE to its true value (default: False)', default=False)
    vb_args.add_argument('-r', '--learning-rate', type=float, help='Initial learning rate for SGD', default=1e-4)
    vb_args.add_argument('-R', '--warmup-rate', type=float, help='Warmup rate for variational regularizer', default=1e-4)
    vb_args.add_argument('-b', '--minibatch-size', type=int, help='Minibatch size for SGD', default=100)
    vb_args.add_argument('-i', '--poll-iters', type=int, help='Polling interval for SGD', default=10000)
    vb_args.add_argument('-t', '--tolerance', type=float, help='Maximum change in objective function (for convergence)', default=1e-4)
    vb_args.add_argument('-w', '--ewma-weight', type=float, help='Exponential weight for SGD objective moving average', default=0.1)

    bootstrap_args = parser.add_mutually_exclusive_group()
    bootstrap_args.add_argument('--resample', action='store_true', help='Resample genotypes to desired sample size')
    bootstrap_args.add_argument('--parametric-bootstrap', type=int, help='Parametric bootstrap trial for frequentist standard errors', default=None)
    bootstrap_args.add_argument('--nonparametric-bootstrap', type=int, help='Nonparametric bootstrap trial for frequentist standard errors', default=None)
    bootstrap_args.add_argument('--validation', type=int, help='Hold out validation set for posterior predictive check', default=None)

    return parser

def _validate(args):
    # Check argument values
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
    elif args.model == 'logistic' and args.study_prop is None:
        args.study_prop = 0.5
        logger.warn('Assuming study prevalence 0.5')
    elif args.study_prop is not None and not 0 <= args.study_prop <= 1:
        raise _A('Case study proportion must be in [0, 1]')
    if args.prevalence is None and args.model == 'logistic':
        raise _A('Prevalence must be specified for logistic model')
    if numpy.isclose(args.min_pve, 0) or args.min_pve < 0:
        raise _A('Minimum PVE must be larger than float tolerance')
    if args.min_maf is not None and (numpy.isclose(args.min_maf, 0) or args.min_maf < 0):
        raise _A('Minimum MAF must be larger than float tolerance')
    if args.max_maf is not None and args.max_maf > .5:
        raise _A('Maximum MAF must be less than or equal to 0.5')
    if args.learning_rate <= 0:
        raise _A('Learning rate must be positive')
    if args.learning_rate > 0.05:
        logger.warn('Learning rate set to {}. This is probably too large'.format(args.learning_rate))
    if args.warmup_rate < 0:
        raise _A('Warmup rate must be non-negative')
    if args.warmup_rate >= 1:
        logger.warn('Warmup rate set to {}. This is probably too large'.format(args.warmup_rate))
    if args.minibatch_size <= 0:
        raise _A('Minibatch size must be positive')
    if args.minibatch_size > args.num_samples:
        logger.warn('Setting minibatch size to sample size')
        args.minibatch_size = args.num_samples
    if args.poll_iters <= 0:
        raise _A('Polling interval must be positive')
    if args.tolerance <= 0:
        raise _A('VB tolerance must be positive')
    if args.wsabi_tolerance <= 0:
        raise _A('Active sampling tolerance must be positive')
    if args.init_samples <= 0:
        raise _A('Number of initial hyperparameter samples must be positive')
    if args.max_samples < args.init_samples:
        raise _A('Maximum number hyperparameter samples must be >= initial number ({})'.format(args.init_samples))
    if not 0 < args.ewma_weight < 1:
        raise _A('Moving average weight must be in (0, 1)')
    if numpy.isclose(args.min_pve, 0) or args.min_pve < 0:
        raise _A('Minimum PVE must be larger than float tolerance')
    max_ = args.num_variants // len(args.annotation)
    for i, (num, var) in enumerate(args.annotation):
        if not 0 <= num <= max_:
            raise _A('Annotation {} must have between 0 and {} causal variants ({} specified)'.format(i, max_, num))
        if var <= 0:
            raise _A('Annotation {} must have positive effect size variance'.format(i))

    # Check if desired method is supported
    if args.method != 'dsvi' and any(k in args for k in ('learning_rate', 'minibatch_size', 'poll_iters', 'ewma_weight', 'warmup_rate')):
        logger.warn('Ignoring SGD parameters for method {}'.format(args.method))
    if args.method in ('varbvs',) and len(args.annotation) > 1:
        raise _A('Method {} does not support multiple annotations'.format(args.method))
    if args.outer_method == 'wsabi' and args.method not in ('coord', 'dsvi'):
        raise _A('Outer method "{}" does not support inner method "{}"'.format(args.outer_method, args.method))
    if args.outer_method != 'wsabi' and 'wsabi_tolerance' in args:
        logger.warn('Ignoring WSABI parameters for method {}'.format(args.outer_method))
    if args.parametric_bootstrap is not None and (args.load_data is not None or args.load_oxstats is not None):
        raise _A('Parametric bootstrap not supported for real data')
    if args.bayes_factor and len(args.annotation) == 1:
        raise _A('Model comparison not supported for only one annotation')
    if args.bayes_factor and args.method in ('varbvs',):
        raise _A('Model comparison not supported for method {}'.format(args.method))
    if args.true_causal and args.method not in ('coord', 'dsvi'):
        raise _A('Fixing causal variants not supported for method {}'.format(args.method))
    if args.propose_tau and args.method not in ('coord', 'dsvi'):
        raise _A('Proposing tau not supported for method {}'.format(args.method))

def evaluate():
    """Entry point for simulations on synthetic genotypes/phenotypes/annotations"""
    args = _parser().parse_args()
    _validate(args)
    logging.getLogger('ctra').setLevel(args.log_level)
    logger.debug('Parsed arguments:\n{}'.format(pprint.pformat(vars(args))))
    with ctra.simulation.simulation(args.num_variants, args.pve, args.annotation, args.seed) as s:
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

        if args.validation is not None:
            args.num_samples += args.validation
        if args.load_data is not None:
            with open(os.path.join(args.load_data, 'genotypes.txt'), 'rb') as f:
                x = numpy.loadtxt(f)
            with open(os.path.join(args.load_data, 'phenotypes.txt'), 'rb') as f:
                y = numpy.loadtxt(f)
        elif args.load_oxstats:
            logger.debug('Loading OXSTATS datasets')
            with contextlib.ExitStack() as stack:
                data = [stack.enter_context(ctra.formats.oxstats_genotypes(*a))
                        for a in ctra.algorithms.kwise(args.load_oxstats, 2)]
                samples = list(itertools.chain.from_iterable(s for _, _, s, _ in data))
                merged = ctra.formats.merge_oxstats([d for _, _, _, d in data])
                probs = ([float(x) for x in row[5:]] for row in merged)
                if args.num_samples > len(samples) and not args.resample:
                    logger.error('{} individuals present in OXSTATS data, but {} were specified'.format(len(samples), args.num_samples))
                    sys.exit(1)
                x = numpy.array(list(itertools.islice(probs, args.num_variants)))
            p, n = x.shape
            if p < args.num_variants:
                logger.error('{} variants present in OXSTATS data, but {} were specified'.format(p, args.num_variants))
                sys.exit(1)
            if not args.resample and args.num_samples < n // 3:
                n = args.num_samples
            else:
                n = n // 3
            x = (x.reshape(p, -1, 3) * numpy.array([0, 1, 2])).sum(axis=2).T[:n,:]
            s.estimate_mafs(x)
            if args.resample:
                logger.debug('Resampling individuals')
                index = s.random.choice(x.shape[0], size=args.num_samples)
                x = x[index,:]
            y = s.compute_liabilities(x)
        else:
            if args.parametric_bootstrap is None:
                # Without bootstrapping, we take the first sample from the
                # generative model
                args.parametric_bootstrap = 0
            else:
                logger.debug('Burning in {} samples for parametric bootstrap sample {}'.format(args.num_samples * args.parametric_bootstrap, args.parametric_bootstrap))
            for _ in range(args.parametric_bootstrap + 1):
                if args.model == 'logistic':
                    x, y = s.sample_case_control(n=args.num_samples, K=args.prevalence, P=args.study_prop)
                else:
                    x, y = s.sample_gaussian(n=args.num_samples)
            if args.nonparametric_bootstrap is not None:
                for _ in range(args.nonparametric_bootstrap):
                    sample = s.random.choice(args.num_samples, args.num_samples)
                x = x[sample,:]
                y = y[sample]
        if args.center or args.normalize:
            x -= x.mean(axis=0)
            y -= y.mean()
        if args.normalize:
            x /= x.var(axis=0)
            y /= y.var()
        if args.rotate:
            logger.info('Computing SVD of genotypes')
            _, v = scipy.linalg.eigh(numpy.inner(x, x))
            rotation = scipy.linalg.inv(v)
            y = rotation.dot(y)
            x = rotation.dot(x)
        if args.write_data is not None:
            if not os.path.exists(args.write_data):
                os.mkdir(args.write_data)
            with open(os.path.join(args.write_data, 'genotypes.txt'), 'wb') as f:
                numpy.savetxt(f, x, fmt='%.3f')
            with open(os.path.join(args.write_data, 'phenotypes.txt'), 'wb') as f:
                numpy.savetxt(f, y, fmt='%.3f')
            with open(os.path.join(args.write_data, 'theta.txt'), 'wb') as f:
                numpy.savetxt(f, s.theta, fmt='%.3f')
            return
        if args.write_sfba_data is not None:
            if not os.path.exists(args.write_sfba_data):
                os.mkdir(args.write_sfba_data)
            with gzip.open(os.path.join(args.write_sfba_data, 'test.geno.gz'), 'wt') as f:
                print('##fileformat=VCFv4.2', file=f)
                print('#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT',
                      '\t'.join('s{}'.format(i) for i in range(x.shape[0])), sep='\t',
                      file=f)
                gt = ('0/0', '0/1', '0/2')
                for i, row in enumerate(x):
                    print('1', i, 'rs{}'.format(i), 'A', 'C', '.', '.', '.', 'GT', '\t'.join(gt[int(x_j)] for x_j in row), sep='\t', file=f)
            with open(os.path.join(args.write_sfba_data, 'phenotypes.txt'), 'wt') as f:
                for i, p in enumerate(y):
                    print('s{}'.format(i), p, file=f)
            with gzip.open(os.path.join(args.write_sfba_data, 'Anno_test.gz'), 'wt') as f:
                print('ID	#CHROM	POS', file=f)
                for i, a in enumerate(args.annotation):
                    print('rs{}'.format(i), '1', i, a, sep='\t', file=f)
                    with open(os.path.join(args.write_sfba_data, 'AnnoCode.txt'), 'w') as f:
                        print('#n_type', len(args.annotation), sep='\t', file=f)
                        for i, a in enumerate(args.annotation):
                            print('a{}'.format(i), i, file=f)
            with open(os.path.join(args.write_sfba_data, 'fileheads.txt'), 'w') as f:
                print('test', file=f)
            with open(os.path.join(args.write_sfba_data, 'hyperparameters.txt'), 'w') as f:
                print('pi', 'sigma2', sep='\t', file=f)
                for a in args.annotation:
                    print(1e-6, 10, file=f)
            return
        if args.validation is not None:
            x_validate = x[-args.validation:]
            y_validate = y[-args.validation:]
            x = x[:-args.validation]
            y = y[:-args.validation]

        if args.true_pve:
            pve = numpy.array([s.genetic_var[s.annot == a].sum() / s.pheno_var
                               for a in range(1 + max(s.annot))])
        else:
            pve = ctra.model.pcgc(y, ctra.model.grm(x, s.annot), K=args.prevalence)
        pve = numpy.clip(pve, args.min_pve, 1 - args.min_pve)
        kwargs = {}
        if args.true_causal:
            kwargs['true_causal'] = ~numpy.isclose(s.theta, 0)
        if args.propose_tau:
            kwargs['propose_tau'] = True
        if args.method == 'pcgc':
            numpy.savetxt(sys.stdout.buffer, pve, fmt='%.3g')
            return
        elif args.method == 'varbvs':
            m = ctra.model.varbvs(x, y, pve, 'multisnphyper' if args.model ==
                                  'gaussian' else 'multisnpbinhyper', **kwargs)
        elif args.method == 'sklearn':
            import sklearn.linear_model
            logger.info('Fitting sklearn model')
            if args.model == 'gaussian':
                m = sklearn.linear_model.ElasticNetCV(l1_ratio=numpy.arange(0, 1, .2),
                                                      fit_intercept=not args.center).fit(x, y)
            else:
                m = sklearn.linear_model.LogisticRegressionCV(fit_intercept=True).fit(x, y)
        else:
            if args.method == 'coord':
                if args.model == 'gaussian':
                    model = ctra.model.GaussianCoordinateAscent
                else:
                    model = ctra.model.LogisticCoordinateAscent
            else:
                if args.model == 'gaussian':
                    model = ctra.model.GaussianDSVI
                else:
                    model = ctra.model.LogisticDSVI
            inner = model(x, y, s.annot, pve, learning_rate=args.learning_rate,
                          warmup_rate=args.warmup_rate,
                          minibatch_n=args.minibatch_size)
            if args.outer_method == 'is':
                outer = ctra.model.ImportanceSampler
            else:
                outer = ctra.model.ActiveSampler
            logger.info('Fitting alternate model')
            m = outer(inner).fit(atol=args.tolerance,
                                 pool=args.pool,
                                 poll_iters=args.poll_iters,
                                 init_samples=args.init_samples,
                                 max_samples=args.max_samples,
                                 vtol=args.wsabi_tolerance,
                                 weight=args.ewma_weight,
                                 **kwargs)
            if args.bayes_factor or args.fit_null:
                logger.info('Fitting null model')
                proposals = numpy.array(m.pi_grid).dot(inner.p) / inner.p.sum()
                inner = model(x, y, numpy.zeros(s.annot.shape, dtype='int8'),
                              pve.sum().reshape(1, 1),
                              learning_rate=args.learning_rate,
                              minibatch_n=args.minibatch_size)
                m0 = outer(inner).fit(atol=args.tolerance, proposals=proposals, **kwargs)
        if args.write_weights is not None:
            logger.info('Writing importance weights:')
            with open(os.path.join(args.write_weights, 'weights.txt'), 'w') as f:
                for p, w in zip(m.pi_grid, m.weights):
                    print('{} {}'.format(' '.join('{:.3g}'.format(x) for x in p),
                                         w), file=f)
        if args.plot:
            figure()
            m.evidence_gp.plot().get_figure().savefig('{}-gp.pdf'.format(args.plot))
            close()

            fig, ax = subplots(2, 1)
            xlabel('Causal variant')
            q = len(s.theta[s.theta != 0])
            ax[0].bar(range(q), s.genetic_var[s.theta != 0])
            ax[0].set_ylabel('Genetic variance')
            ax[1].bar(range(q), m.pip[s.theta != 0])
            ax[1].set_ylabel('PIP')
            savefig('{}-pip.pdf'.format(args.plot))
            close()
        if args.validation is not None:
            logger.info('Training set correlation = {:.3f}'.format(m.score(x, y)))
            logger.info('Validation set correlation = {:.3f}'.format(m.score(x_validate, y_validate)))
        if args.bayes_factor:
            logger.info('Bayes factor = {:.3g}'.format(m.bayes_factor(m0)))
        if args.interact:
            code.interact(banner='', local=dict(globals(), **locals()))
        if args.fit_null:
            m = m0
        logger.info('Writing posterior mean pi')
        numpy.savetxt(sys.stdout.buffer, m.pi, fmt='%.3g')
