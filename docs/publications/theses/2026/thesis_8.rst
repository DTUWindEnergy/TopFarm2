.. thesis_8:

Wind farm optimization with computationally efficient AEP models.
===============================================================================================================

**Abstract**

As wind farms expand to meet energy transition goals, wake effects create significant
challenges, giving rise to the Wind Farm Layout Optimization (WFLO) problem. Current
methods to maximize Annual Energy Production (AEP) by minimizing wakes are
computationally intensive, requiring hundreds of iterations and multiple optimizations
per project. Several AEP models with a variety of simplifications, assumptions and approaches,
were tested as objective functions in a gradient based optimization on a complex
167 turbine case study, both with and without grid constraints, alongside other constraints
such as minimum turbine distance, boundaries, and exclusion zones. The FLOW Estimation
and Rose Superposition (FLOWERS) method emerged as the most effective, balancing
efficiency and efficacy with the lowest wake losses and optimization times, thanks to
its fast evaluations, smooth design space – coming from the continuous form using Fourier
– and single CPU configuration. Stochastic Gradient Descent (SGD), thanks to its algorithm
and stochastic nature, performed greatly in terms of AEP, with a performance
close to FLOWERS, accomplishing fair efficiency only when using a reduced number of
iterations, despite its loss in efficacy. Although complex models with many speed and direction
bins delivered consistent results, they required 5-6 times more computational time
despite using 8 CPUs. On the other hand, simpler models, such as those using average
wind speed or a uniform thrust coefficient, proved highly efficient but poorly reflected
the physics, leading to suboptimal layouts. In addition, it was concluded that models
with fewer wind directions and Gaussian-shaped wake models effectively captured layout
changes, whereas those averaging wind speed per direction missed critical physical
dynamics. Overall, FLOWERS outperformed all other models and demonstrated strong
potential for other optimization problems applications, including nested and multi-stage
optimizations.

**Cite this**

Bueno, Javier Andueza. (2026). Wind farm optimization with computationally efficient AEP models.

**Link**

Download `here
<https://findit.dtu.dk/en/catalog/69c72a7bcbf51832fcad2c28>`_.

.. tags:: Thesis, Layout Optimization, AEP models, FLOWERS