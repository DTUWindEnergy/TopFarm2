.. Example 2

Example 2: Optimization with Different Wake Models
===================================================

This example demonstrates the optimization of a 3-turbine farm with
the N. O. Jensen (NOJ) and G. C. Larsen (GCL) wake models implemented in
FUSED-Wake.

Specifications
--------------

- The farm is offshore.  
- The wind resource is defined by arrays of the frequency and Weibull
  A and k parameters taken from a Horns Rev calculation.  
- There is a boundary beyond which the turbines cannot go.  
- The turbines cannot be closer than 2D.

Results
-------

The optimization results are visualized in the figure below, and the
script also outputs the following information on the resultings costs::

   Comparison of cost models vs. optimized locations:
   
   Cost    |    GCL_aep      NOJ_aep
   ---------------------------------
   GCL_loc |   -2704.40    -2688.57   (0.59%)
   NOJ_loc |   -2703.90    -2688.59   (0.57%)
                (0.02%)     (-0.00%)

The subscript ``_aep`` indicates the AEP calculated using the specified wake
model. The subscript ``_loc`` indicates the optimal locations determined when
optimizing with the specified wake model. For example, the cost corresponding
to the ``GCL_aep`` column and the ``NOJ_loc`` column is the AEP calculated
using the GCL wake model for the optimal location results when the N. O.
Jensen wake model was used in the optimization. The percentages in parentheses
are the percent differences for the column or row.  

Important observations:  

- The upper and lower turbines are pushed away from each other, towards
  the corners.  
- There is little difference in the AEP value when calculated using either of
  the optimal locations, since the optimal locations themselves are quite
  close.  
- The N. O. Jensen wake model has an AEP that is approximately 0.6% lower than
  the AEP with the G. C. Larsen wake model for this farm.  

.. figure:: /../../examples/example_2_wake_comparison.png

Code
----

.. literalinclude:: /../../examples/example_2_wake_comparison.py
