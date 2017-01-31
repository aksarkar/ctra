import setuptools

setuptools.setup(
    name='ctra',
    description='Complex Trait Regulatory Architectures',
    version='0.6',
    url='https://github.mit.edu/aksarkar/ctra',
    author='Abhishek Sarkar',
    author_email='aksarkar@mit.edu',
    license='BSD',
    install_requires=['matplotlib', 'numpy', 'scipy', 'sklearn', 'theano', 'GPy'],
    entry_points={
        'console_scripts': [
            'ctra-evaluate=ctra.evaluate:evaluate',
        ]
    }
)
