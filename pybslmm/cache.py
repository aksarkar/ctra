import contextlib
import hashlib
import os
import os.path
import pickle

import pystan

@contextlib.contextmanager
def model(file=None, model_code=None, model_name=None, **kwargs):
    cachedir = os.path.join(os.path.dirname(__file__), 'cached_models')
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)
    if file is not None:
        with open(file, 'rb') as f:
            data = f.read()
    elif model_code is not None:
        data = model_code.encode()
    else:
        raise ArgumentError("One of file or model_code must be provided")
    key = os.path.join(cachedir, '{}'.format(hashlib.md5(data).hexdigest()))
    hit = False
    try:
        with open(key, 'rb') as f:
            m = pickle.load(f)
            hit = True
    except:
        m = pystan.StanModel(file=file, model_code=model_code)
    try:
        yield m
    except Exception as e:
        if not hit:
            with open(key, 'wb') as f:
                m = pickle.dump(m, f)
        raise e
