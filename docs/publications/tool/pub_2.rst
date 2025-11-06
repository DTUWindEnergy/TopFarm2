.. pub_2:

TopFarm2: Plant-level optimization tool developed by DTU Wind and Energy Systems.
==========================================================================================================================================

**Abstract**

TOPFARM is a Python package developed by DTU Wind Energy that serves as a wind farm optimizer for both onshore and offshore wind farms. It uses the OpenMDAO package for optimization and wraps the PyWake package for easy computation of a wind farm’s Annual Energy Production (AEP).

In addition, it can compute other metrics such as the Internal Rate of Return (IRR) and Net Present Value (NPV) and utilizes different engineering wake models available in PyWake to perform the flow simulations.

Over the years, TOPFARM has become a highly versatile tool that is capable of solving several types of optimization problems with different design variables and objectives functions in mind. Throughout its development, TOPFARM has evolved from simple layout optimization problems to more complex and relevant wind farm optimization scenarios. Its capabilities and range were designed for both research and industry related topics. Today, TOPFARM can provide the user solutions in:

Wind farm layout optimization for different turbine types

Wind farm layout optimization for different turbine hub heights

Active control (wake steering) optimization

Load constrained layout optimization

Load constrained wake steering optimization

Optimization with bathymetry

LCoE-based layout optimization

Additionally, the objective function in TOPFARM can be formulated in economical terms, that is with the inclusion of several financial factors that are inherent in the wind farm design process. These can include the financial balance, foundation costs, electrical costs (cabling), fatigue degradation of turbine components and Operation and Management (O&M) costs.

The calculations for the wind farm interactions are done through PyWake, which is responsible for computing the wake losses and power production of both individual turbines and whole wind farms with the use of engineering wake models. In TOPFARM, the objective function is evaluated by the cost model component, and can be represented by either power production or financial goals.

TOPFARM comes with many built-in wake and cost models that were designed to accurately represent the optimization problem at hand. However, the tool is very flexible, and users are also able to perform custom optimizations as well.

For installation instructions, please see the Installation Guide. The base code is open-source and freely available on GitLab (MIT license).

**Cite this**

Riccardo Riva, Jaime Yikon Liew, Mikkel Friis-Møller, Nikolay Krasimirov Dimitrov, Emre Barlas, Pierre-Elouan Réthoré, m.fl. TopFarm2. Zenodo; 2019

**Link**

Download `here
<https://topfarm.pages.windenergy.dtu.dk/TopFarm2/#>`_.

.. tags:: Layout Optimization, Control Optimization