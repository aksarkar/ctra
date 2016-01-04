import setuptools

setuptools.setup(
    name='pybslmm',
    description='Python implementation of Bayesian Sparse Linear Mixed Models',
    url='https://github.mit.edu/aksarkar/pybslmm',
    author='Abhishek Sarkar',
    author_email='aksarkar@mit.edu',
    license='BSD',
    install_requires=['numpy', 'pystan'],
    package_data={'pybslmm': ['*.stan']}
)
