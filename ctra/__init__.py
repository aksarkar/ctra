import logging
import os

import numpy

# This is an ugly hack to get around Univa Grid Engine nonsense
if 'LD_LIBRARY_PATH' not in os.environ and 'LIBRARY_PATH' in os.environ:
    logging.getLogger('ctra').debug('Setting LD_LIBRARY_PATH={}'.format(os.getenv('LIBRARY_PATH')))
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LIBRARY_PATH')

import ctra.experiments
import ctra.model
import ctra.simulation

logging.basicConfig(format='[%(asctime)s %(name)s] %(message)s', level=logging.DEBUG)

