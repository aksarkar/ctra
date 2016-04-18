import setuptools

setuptools.setup(
    name='pybslmm',
    description='Python implementation of Bayesian Sparse Linear Mixed Models',
    version='0.1',
    url='https://github.mit.edu/aksarkar/pybslmm',
    author='Abhishek Sarkar',
    author_email='aksarkar@mit.edu',
    license='BSD',
    install_requires=['numpy', 'scipy', 'theano']
)
