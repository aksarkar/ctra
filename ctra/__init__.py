import logging
import os

import numpy

if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LIBRARY_PATH')

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)
numpy.seterrcall(lambda err, flag: logging.warn(err))
numpy.seterr(all='call')
