# -*- coding: utf-8 -*-
"""
Setup file for Topfarm2
"""


import os
from git_utils import write_vers
from setuptools import setup, find_packages

repo = os.path.dirname(__file__)
version = write_vers(vers_file='topfarm/__init__.py', repo=repo, skip_chars=1)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='topfarm',
      version=version,
      description='Topfarm - Wind farm optimization using OpenMDAO',
      long_description=read('README'),
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/topfarm2',
      author='DTU Wind Energy',
      author_email='mikf@dtu.dk',
      license='MIT',
      packages=find_packages(),
      install_requires=[
              'matplotlib',  # for plotting
              'numpy',  # for numerical calculations
              'numpy-financial',  # for irr calculations
              'openmdao>=3.16',  # for optimization
              'pytest',  # for testing
              'pycodestyle',  # for testing
              'pytest-cov',  # for calculating coverage
              'py_wake>=2',  # for calculating AEP
              'pyDOE2',  # for GA driver
              'pymongo[srv]',  # for mongo_recorder
              'scipy',  # constraints
              'sphinx',  # generating documentation
              'sphinx_rtd_theme',  # docs theme
              'scikit-learn',  # load surrogate
              'mock',  # replace variables during tests
              'tensorflow',  # loads examples with surrogates
              'shapely>=1.8.2',  # for multiple polygon boundaries and exclusion zones
              'memory_profiler',  # memory profiling
      ],
      zip_safe=True)
