"""Call into varbvs Matlab implementation

This is the simplest possible implementation: just write the data out to disk
and read it back into a Matlab subprocess.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import logging
import os.path
import subprocess
import tempfile

import numpy

matlab_code = """
x = dlmread('{data}/genotypes.txt');
y = dlmread('{data}/phenotype.txt');
[h theta0] = ndgrid({pve}, -3:.5:0);
[logw alpha mu s eta] = {function}(x, y, h, theta0);
w = normalizelogweights(logw);
sigmoid10(dot(w, theta0))
"""

logger = logging.getLogger(__name__)

_result = collections.namedtuple('result', ['pi'])

def varbvs(x, y, pve, function):
    """Return the output of the Matlab coordinate ascent implementation"""
    with tempfile.TemporaryDirectory() as data:
        logger.info('Writing data to temporary file')
        numpy.savetxt(os.path.join(data, 'genotypes.txt'), x, fmt='%.3f')
        numpy.savetxt(os.path.join(data, 'phenotype.txt'), y, fmt='%.3f')
        matlab_args = {'data': data,
                       'function': function,
                       'pve': pve.ravel()[0]}
        logger.info('Starting Matlab subprocess')
        with subprocess.Popen(['matlab', '-nodesktop'], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            out, err = p.communicate(bytes(matlab_code.format(**matlab_args), 'utf-8'))
            out = str(out, 'utf-8').split('\n')
            for line in out:
                logger.debug(line)
            if err:
                print(str(err, 'utf-8'))
                raise RuntimeError
            return _result(numpy.array([float(out[-3])]))
