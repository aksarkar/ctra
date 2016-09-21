"""Evaluate the accuracy of the model

We want to recover the correct pi and tau, and also the correct posterior on
theta and z. We use the distribution of parameter estimates in simulated
training sets for the first, and AUPRC on a simulated validation set for the
second.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import argparse
import collections
import logging
import pprint
import sys

import numpy

import ctra.pcgc
import ctra.model
import ctra.simulation

logger = logging.getLogger(__name__)

def evaluate():
    """Entry point for simulations on synthetic genotypes/phenotypes/annotations"""
    _A = argparse.ArgumentTypeError

    def Annotation(arg):
        num, var = arg.split(',')
        return (int(num), float(var))

    annotation = lambda arg: tuple([f(x) for f, x in zip([int, float], arg.split())])
    parser = argparse.ArgumentParser(description='Evaluate the model on synthetic data')
    parser.add_argument('-a', '--annotation', type=Annotation, action='append', help="""Annotation parameters (num. causal, effect size var.) separated by ','. Repeat for additional annotations.""", default=[], required=True)
    parser.add_argument('-n', '--num-samples', type=int, help='Number of samples', default=1000)
    parser.add_argument('-p', '--num-variants', type=int, help='Number of genetic variants', default=10000)
    parser.add_argument('-v', '--pve', type=float, help='Total proportion of variance explained', default=0.25)
    parser.add_argument('--true-pve', action='store_true', help='Fix hyperparameter PVE to its true value (default: False)', default=False)
    parser.add_argument('--min-pve', type=float, help='Minimum PVE', default=1e-4)
    parser.add_argument('-K', '--prevalence', type=float, help='Binary phenotype case prevalence (assume Gaussian if omitted)', default=None)
    parser.add_argument('-P', '--study-prop', type=float, help='Binary phenotype case study proportion (assume 0.5 if omitted but prevalence given)', default=None)
    parser.add_argument('-m', '--model', choices=['pcgc', 'gaussian', 'logistic'], help='Type of model to fit')
    parser.add_argument('-r', '--learning-rate', type=float, help='Initial learning rate for SGD', default=1e-4)
    parser.add_argument('-b', '--minibatch-size', type=int, help='Minibatch size for SGD', default=100)
    parser.add_argument('-i', '--poll-iters', type=int, help='Polling interval for SGD', default=10000)
    parser.add_argument('-w', '--ewma-weight', type=float, help='Exponential weight for SGD objective moving average', default=0.1)
    parser.add_argument('-s', '--seed', type=int, help='Random seed', default=0)
    parser.add_argument('-l', '--log-level', choices=['INFO', 'DEBUG'], help='Log level', default='INFO')
    args = parser.parse_args()

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
    elif args.study_prop is None:
        args.study_prop = 0.5
        logger.warn('Assuming study prevalence 0.5')
    elif not 0 <= args.study_prop <= 1:
        raise _A('Case study proportion must be in [0, 1]')
    if args.prevalence is None and args.model == 'logistic':
        raise _A('Prevalence must be specified for logistic model')
    if args.model == 'gaussian' and any(k in args for k in ('learning_rate', 'minibatch_size', 'poll_iters', 'ewma_weight')):
        logger.warn('Ignoring SGD parameters for Gaussian model')
    if not args.annotation:
        raise _A('Must specify at least one annotation')
    else:
        max_ = args.num_variants // len(args.annotation)
        for i, (num, var) in enumerate(args.annotation):
            if not 0 <= num <= max_:
                raise _A('Annotation {} must have between 0 and {} causal variants ({} specified)'.format(i, max_, num))
            if var <= 0:
                raise _A('Annotation {} must have positive effect size variance'.format(i))
    if numpy.isclose(args.min_pve, 0) or args.min_pve < 0:
        raise _A('Minimum PVE must be larger than float tolerance')

    logging.getLogger('ctra').setLevel(args.log_level)
    logging.info('Parsed arguments:\n{}'.format(pprint.pformat(vars(args))))

    with ctra.simulation.simulation(args.num_variants, args.pve, args.annotation, args.seed) as s:
        if args.prevalence is not None:
            x, y = s.sample_case_control(n=args.num_samples, K=args.prevalence, P=args.study_prop)
        else:
            x, y = s.sample_gaussian(n=args.num_samples)
        if args.true_pve:
            pve = numpy.array([s.genetic_var[s.annot == a].sum() / s.pheno_var for a in range(1 + max(s.annot))])
        else:
            pve = ctra.pcgc.estimate(y, ctra.pcgc.grm(x, s.annot), K=args.prevalence)
        pve = numpy.clip(pve, args.min_pve, 1 - args.min_pve)
        if args.model == 'pcgc':
            numpy.savetxt(sys.stdout.buffer, pve, fmt='%.3e')
            return
        elif args.model == 'gaussian':
            m = ctra.model.GaussianModel(x, y, s.annot, pve).fit()
        else:
            m = ctra.model.LogisticModel(x, y, s.annot, K=args.prevalence,
                                         pve=pve,
                                         learning_rate=args.learning_rate,
                                         minibatch_n=args.minibatch_size)
            m.fit(poll_iters=args.poll_iters, weight=args.ewma_weight)
        logging.info('Writing posterior mean pi')
        numpy.savetxt(sys.stdout.buffer, m.pi, fmt='%.3g')
