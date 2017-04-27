import setuptools

setuptools.setup(
    name='ctra',
    description='Complex Trait Regulatory Architectures',
    version='0.7',
    url='https://github.mit.edu/aksarkar/ctra',
    author='Abhishek Sarkar',
    author_email='aksarkar@mit.edu',
    license='MIT',
    install_requires=['h5py', 'matplotlib', 'numpy', 'robo', 'scipy', 'theano'],
    dependency_links = [
        'https://github.com/automl/robo/tarball/master#egg=robo'
    ]
    entry_points={
        'console_scripts': [
            'ctra-evaluate=ctra.evaluate:evaluate',
            'ctra-convert=ctra.convert:convert'
        ]
    }
)
