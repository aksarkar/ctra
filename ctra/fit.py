"""Fit the model on real data

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import argparse
import contextlib
import gzip
import itertools
import pickle
import logging
import sys

import numpy as np
import sklearn.metrics
import pyplink

import ctra

logger = logging.getLogger(__name__)
_A = argparse.ArgumentTypeError

def _parser():
    parser = argparse.ArgumentParser(description='Fit the model')
    req_args = parser.add_argument_group('Required arguments', '')
    req_args.add_argument('-m', '--model', choices=['gaussian', 'logistic', 'bslmm'], help='Type of model to fit', required=True)
    req_args.add_argument('-n', '--num-samples', type=int, help='Number of samples', required=True)
    req_args.add_argument('-p', '--num-variants', type=int, help='Number of genetic variants', required=True)
    req_args.add_argument('-v', '--validation', type=int, help='Hold out validation set size', required=True)

    input_args = parser.add_mutually_exclusive_group()
    input_args.add_argument('-P', '--load-plink', help='Plink data set (prefix)', default=None)
    input_args.add_argument('-G', '--load-oxstats', nargs='+', help='OXSTATS data sets (.sample, .gen.gz)', default=None)
    input_args.add_argument('-H', '--load-hdf5', help='HDF5 data set', default=None)

    parser.add_argument('--pheno', help='Phenotype column name', default='pheno')
    parser.add_argument('-b', '--bootstrap', action='store_true', help='Bootstrap the samples', default=False)
    parser.add_argument('-A', '--annotation-matrix', help='Annotation matrix')
    parser.add_argument('-l', '--log-level', choices=['INFO', 'DEBUG'], help='Log level', default='INFO')
    parser.add_argument('-o', '--output', help='Output file for pickled result', default=None)
    return parser

def _validate(args):
    """Validate command line arguments"""
    if args.num_samples <= 0:
        raise _A('Number of samples must be positive')
    if args.num_variants <= 0:
        raise _A('Number of variants must be positive')
    if args.validation <= 0:
        raise _A('Number of held-out samples must be positive')

def _load_data(args):
    result = {}

    if args.annotation_matrix is not None:
        logger.debug('Loading annotation matrix')
        with gzip.open(args.annotation_matrix, 'rt') as f:
            result['a'] = np.loadtxt(f).astype('int8')
        if a.shape[0] != args.num_variants:
            raise _A('{} variants present in annotations file, but {} specified'.format(a.shape[0], args.num_variants))
    else:
        result['a'] = np.ones((args.num_variants, 1))

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
            x = np.array(list(itertools.islice(probs, args.num_variants)))
        p, n = x.shape
        if p < args.num_variants:
            logger.error('{} variants present in OXSTATS data, but {} were specified'.format(p, args.num_variants))
            sys.exit(1)
        n = n // 3
        x = (x.reshape(p, -1, 3) * np.array([0, 1, 2])).sum(axis=2).T[:n,:]
        raise NotImplementedError("Load y")
    elif args.load_hdf5:
        with h5py.File(args.load_hdf5) as f:
            logger.debug('Loading HDF5 dataset')
            x = f['dosage'][:args.num_samples, :args.num_variants]
        raise NotImplementedError("Load y")
    elif args.load_plink:
        with pyplink.PyPlink(args.load_plink) as f:
            if args.num_samples > f.get_nb_samples():
                logger.error('{} individuals present in Plink data, but {} were specified'.format(f.get_nb_samples(), args.num_samples))
                sys.exit(1)
            if args.num_variants > f.get_nb_markers():
                logger.error('{} variants present in Plink data, but {} were specified'.format(f.get_nb_markers(), args.num_variants))
                sys.exit(1)
            logger.debug('Loading Plink dataset')
            x = np.zeros((args.num_samples, args.num_variants), dtype='float32')
            for i, (_, geno) in enumerate(f):
                if i >= args.num_variants:
                    break
                x[:,i] = geno[:args.num_samples].astype('float32')
            # Mask missing genotypes before centering
            x = np.ma.masked_equal(x, -1)
            y = f.get_fam()[args.pheno].values[:args.num_samples] - 1
    # Hold out samples
    if args.model == 'logistic':
        # Permute the data so minibatches/hold out are balanced in expectation
        np.random.shuffle(x)
    # Assume samples are exchangeable
    x_validate = x[-args.validation:]
    y_validate = y[-args.validation:]
    x = x[:-args.validation]
    y = y[:-args.validation]

    # Final pass of the data
    x -= x.mean(axis=0)
    x_validate -= x_validate.mean(axis=0)
    if args.model != 'logistic':
        y -= y.mean()
        y_validate -= y_validate.mean()
    # Replace missing genotypes with the mean (0 after genotyping)
    # This follows e.g. Bolt-LMM (Roh et al 2015)
    if np.ma.isMaskedArray(x):
        x = x.filled(0)
    if np.ma.isMaskedArray(x_validate):
        x_validate = x_validate.filled(0)
    result.update({'x': x, 'y': y, 'x_validate': x_validate, 'y_validate': y_validate})
    return result

class ModelFit(dict):
    """Wrapper around dict to make saving/writing model results cleaner"""
    def __init__(self, prefix, *args, **kwargs):
        self.prefix = prefix
        super().__init__(*args, **kwargs)
        
    def __setitem__(self, k, v):
        super().__setitem__('{}_{}'.format(self.prefix, k), v)

    def __repr__(self):
        return '\n'.join(['{} = {:.3f}'.format(k, v) for k, v in self.items()])

def _fit(args, x, y, x_validate, y_validate, a):
    result = {'args': args}
    model = {'gaussian': ctra.model.GaussianSGVB,
             'logistic': ctra.model.LogisticSGVB,
             'bslmm': ctra.model.VBSLMM
    }[args.model]

    def fit(weights, m0=None):
        if m0 is None:
            result = ModelFit('m0')
        else:
            result = ModelFit('m1')
        m = model(x, y, a, m0=m0, weights=weights, stoch_samples=50,
                  learning_rate=0.1, rho=0.9)
        m.fit(max_epochs=400, xv=x_validate, yv=y_validate)
        result['theta'] = m.theta
        result['pip'] = m.pip
        result['b'] = m.b
        result['c'] = m.c
        result['training_set_score'] = np.asscalar(m.score(x, y))
        result['validation_set_score'] = np.asscalar(m.score(x_validate, y_validate))
        if getattr(m, 'predict_proba', None) is not None:
            auprc = sklearn.metrics.average_precision_score
            result['training_set_auprc'] = auprc(y, m.predict_proba(x))
            result['validation_set_auprc'] = auprc(y_validate, m.predict_proba(x_validate))
        return m, result

    if args.bootstrap:
        weights = np.random.multinomial(args.num_samples, [1 / args.num_samples] * args.num_samples)
    else:
        weights = None

    logger.info('Fitting genome-wide model')
    m0, m0_fit = fit(weights=weights)
    result.update(m0_fit)

    if args.annotation_matrix is not None:
        logger.info('Fitting annotation model')
        m1, m1_fit = fit(weights=weights, m0=m0)
        result.update(m1_fit)

    if args.output is not None:
        with open(args.output, 'wb') as f:
            pickle.dump(f)
    print(result)
    

def main():
    args = _parser().parse_args()
    _validate(args)
    logging.getLogger('ctra').setLevel(args.log_level)
    args.num_samples += args.validation
    data = _load_data(args)
    _fit(args, **data)

if __name__ == '__main__':
    main()
