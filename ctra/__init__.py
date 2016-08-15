import os

if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LIBRARY_PATH')
