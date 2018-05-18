'''
Created on 17. maj 2018

@author: mmpe
'''
import unittest
import topfarm
import pkgutil
import importlib
import inspect
import ast
import warnings
from topfarm.cost_models.fuga import lib_reader
import mock
import pytest
import os


class TestTryMe(unittest.TestCase):

    
    def test_try_me(self):
        if os.name == 'posix' and "DISPLAY" not in os.environ:
            pytest.xfail("No display")
        package = topfarm
        for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = importlib.import_module(modname)
            if 'try_me' in dir(m):
                print("Checking %s.try_me" % modname)
                with mock.patch.object(m, "__name__", "__main__"):
                    getattr(m, 'try_me')()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
