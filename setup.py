# -*- coding: utf-8 -*-
"""
Setup file for Topfarm2
"""


from setuptools import setup

setup(name='topfarm', 
      version='1.0',  
      description='Topfarm - Wind farm optimization using OpenMDAO',
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/topfarm2',
      author='MMPE, PIRE, MIKF, RINK',  
      author_email='mmpe@dtu.dk, pire@dtu.dk, mikf@dtu.dk, rink@dtu.dk',
      license='GNU GPL',
      packages=['constraint_components',
				'cost_models',
				'example_data',
				'tests',
				],
	  files= ['plotting.py',
			  'topfarm.py',
			  ],
      zip_safe=True)
