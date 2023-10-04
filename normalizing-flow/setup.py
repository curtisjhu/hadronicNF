from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use GAN to generate particle physics events"

setup(
    name="gan4hep",
    version="0.3.0",
    description=description,
    long_description=description,
    author="Xiangyang Ju",
    license="Apache License, Version 2.0",
    keywords=["NF", "HEP"],
    url="https://github.com/xju2/gan4hep",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[        
        'tensorflow >= 2.0.0',
        'tensorflow-probability',
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'seaborn',
        'pyyaml>=5.1',
        'uproot',
        'vector',
        'awkward',
        'tables',
        'more_itertools'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ]
)
