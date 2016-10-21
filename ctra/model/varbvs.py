"""Call into varbvs Matlab implementation

This is the simplest possible implementation: just write the data out to disk
and read it back into a Matlab subprocess.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import logging
import os
import os.path
import subprocess
import tempfile

import numpy

from .base import result

logger = logging.getLogger(__name__)

def varbvs(x, y, pve, function, *args):
    """Return the output of the Matlab coordinate ascent implementation"""
    if 'MCRROOT' not in os.environ:
        raise RuntimeError('Method varbvs requires environment variable MCRROOT to be set')
    else:
        root = os.getenv('MCRROOT')
    with tempfile.TemporaryDirectory(dir='/local/scratch') as data:
        logger.info('Writing data to temporary file')
        numpy.savetxt(os.path.join(data, 'genotypes.txt'), x, fmt='%.3f')
        numpy.savetxt(os.path.join(data, 'phenotypes.txt'), y, fmt='%.3f')
        logger.info('Starting Matlab subprocess')
        command = ['run_varbvs.sh', root, data,
                   '{:.3f}'.format(pve.ravel()[0]),
                   function] + [str(a) for a in args]
        logger.debug(str(command))
        with subprocess.Popen(command, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as p:
            out, err = p.communicate()
            ret = p.returncode
        if ret != 0 or err:
            if err:
                for line in str(err, 'utf-8').split('\n'):
                    logger.error(line)
            raise RuntimeError('Matlab process exited with an error')
        for line in str(out, 'utf-8').split('\n'):
            logger.debug(line)
        if function == 'bvsmcmc':
            weights = None
            params = None
        else:
            weights = numpy.loadtxt(os.path.join(data, 'weights.txt'))
            alpha = numpy.loadtxt(os.path.join(data, 'alpha.txt')).T
            beta = numpy.loadtxt(os.path.join(data, 'mu.txt')).T
            params = list(zip(alpha, beta))
        return result(pi=numpy.loadtxt(os.path.join(data, 'pi.txt'), ndmin=1),
                      pi_grid=numpy.arange(-3, 0.25, 0.25).reshape(-1, 1),
                      weights=weights, params=params)
