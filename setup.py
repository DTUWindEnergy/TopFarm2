# -*- coding: utf-8 -*-
"""
Setup file for Topfarm2
"""


from setuptools import setup

setup(name='topfarm', 
      version='2.0',  
      description='Topfarm - Wind farm optimization using OpenMDAO',
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/topfarm2',
      author='DTU Wind Energy',  
      author_email='dave@dtu.dk',
      license='MIT',
      packages=['topfarm'
                ],
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
