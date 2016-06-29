import setuptools

setuptools.setup(
    name='pybslmm',
    description='Python implementation of Bayesian Sparse Linear Mixed Models',
    url='https://github.mit.edu/aksarkar/pybslmm',
    version='0.3',
    author='Abhishek Sarkar',
    author_email='aksarkar@mit.edu',
    license='BSD',
    install_requires=['drmaa', 'numpy', 'scipy', 'theano']
)
