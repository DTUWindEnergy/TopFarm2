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
      author_email='dave@dtu.dk',
      license='MIT',
      packages=find_packages(exclude=["*.examples", "*.examples.*", "examples.*", "examples"]),
      install_requires=[
        'matplotlib',  # for plotting
        'numpy',  # for numerical calculations
        'openmdao==2.3.1',  # for optimization
        'networkx==2.1', # for avoiding a warning/bug
        'pytest',  # for testing
        'pytest-cov',  # for calculating coverage
        'scipy',  # constraints
        'sphinx',  # generating documentation
        'sphinx_rtd_theme',  # docs theme
		'py_wake'
      ],
      zip_safe=True)
