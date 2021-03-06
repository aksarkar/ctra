import setuptools

setuptools.setup(
    name='ctra',
    description='Complex Trait Regulatory Architectures',
    version='0.9',
    url='https://github.mit.edu/aksarkar/ctra',
    author='Abhishek Sarkar',
    author_email='aksarkar@mit.edu',
    license='MIT',
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'pandas',
        'pyplink',
        'scipy',
        'sklearn',
        'theano',
    ],
    entry_points={
        'console_scripts': [
            'ctra-evaluate=ctra.evaluate:evaluate',
            'ctra-fit=ctra.fit:main',
            'ctra-convert=ctra.convert:convert'
        ]
    },
    packages=setuptools.find_packages(),
)
