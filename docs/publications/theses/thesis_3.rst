.. thesis_3:

Wind farm production maximization using combined layout and wake steering optimization under load constraint
==========================================================================================================================

**Abstract**

In recent years, research on wake steering has proven beneficial for increasing the power
production of individual turbines as well as the Annual Energy Production (AEP) of a wind
farm. By having an intentional yaw offset, the wake in the wind farm can be redirected
so that the turbines downstream can capture more energy. However, misalignment of the
turbines with the wind direction and change of their positions within the wind farm comes
with a considerable impact on the fatigue loading for blade root and tower top loads. In
addition, layout optimization by itself has been studied extensively and its benefits are more
than established. Nevertheless, a combined scenario where both the layout and the turbines’
yaw angles are optimized while restricting the loads has not been explored. Thus, this thesis
project aims to investigate how a coupled strategy of wake steering and layout optimization
under load constraints influences a wind farm’s AEP and to explore the likely trade-off between
load and power when operating the turbines under yaw misalignment. For this purpose, an
already trained surrogate model of the DTU 10MW reference turbine is used to perform both
the Damage Equivalent Loads (DEL) and the Lifetime Damage Equivalent Loads (LDEL)
calculation via PyWake’s flow model, and TOPFARM is used to carry out the optimizations.
For the layout optimization, the design variables were chosen as the inter-turbine spacing and
grid rotation angle to reduce the complexity of the problem. A simple 9 turbine wind farm case
is used for the study and a comparison is made between two optimization approaches: integrated
and sequential. The analysis shows that an integrated optimization is highly computational
expensive, even when the design variables for layout optimization are reduced to only two.
On the other hand, the sequential approach yielded a higher AEP gain in substantially less
time, which begs to question whether a fully integrated optimization is justifiable. For both
optimization approaches a uniform distribution of the LDEL was obtained and the constraints
were never reached, which means that further restriction on the maximum allowable loading
could be explored.



**Cite this**

Araujo, Maria Virginia Sarcos, 2022

**Link**

Download `here
<https://findit.dtu.dk/en/catalog/6306bdc6f0d78a1325f51643>`_.

.. tags:: Thesis, Control Optimization, Layout Optimization, Wake Steering, Load Constraint