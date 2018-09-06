.. Example 3

Example 3: Turbine Type Optimization
===================================================

This example uses a dummy cost function to optimize a simple wind turbine
layout that is subject to constraints. The optimization pushes the wind turbine
locations to specified locations in the farm.

Specifications
--------------
- This optimization uses the FullFactorialGenerator from openmdao's doe_generators

Results
-------

The optimization results are visualized in the below GIF.  

.. figure:: /../../examples/example_3_turbine_type_optimization.gif

Code
----

.. literalinclude:: /../../examples/example_3_turbine_type_optimization.py
