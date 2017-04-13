import logging
import os

import numpy

import ctra.model
import ctra.simulation

if 'LD_LIBRARY_PATH' not in os.environ and 'LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LIBRARY_PATH')

logging.basicConfig(format='[%(asctime)s %(name)s] %(message)s', level=logging.DEBUG)
