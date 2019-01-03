# -*- coding: utf-8 -*-
"""
Setup file for Topfarm2
"""


import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='topfarm', 
      version='2.0.3',  
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
        'pytest',  # for testing
        'pytest-cov',  # for calculating coverage
        'scipy',  # constraints
        'sphinx',  # generating documentation
        'sphinx_rtd_theme'  # docs theme
      ],
      zip_safe=True)
