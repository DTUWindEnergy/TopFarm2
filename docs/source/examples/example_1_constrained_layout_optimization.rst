.. Example 1

Example 1: Constrained Layout Optimization
===========================================

This example demonstrates the optimization of a wind turbine layout that is
subject to boundary constraints.

Specifications
--------------

- The cost function is a dummy function that penalizes the turbines when they
  are far away from pre-specified, desired positions.  
- There is a boundary beyond which the turbines cannot go.  
- The turbines cannot be closer than 2D.

Results
-------

The optimization results are visualized in the figure below. The turbines'
trajectories during the optimizations are shown by the colored lines.

- All turbines that begin outside the boundary immediately jump inside the
  boundary.  
- Turbines 1 and 3 rotate around each other as they try to reach their
  specified final locations but stay at least 2D away from each other.
- Turbine 2 remains on the boundary but tries to minimize its distance to
  its specified final location.  
- Turbine 4 converges to its specified location.

.. figure:: /../../examples/docs/figures/example_1_constrained_layout_optimization.png

Code
----

.. literalinclude:: /../../examples/docs/example_1_constrained_layout_optimization.py
