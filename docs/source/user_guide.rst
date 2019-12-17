.. _user_guide:

User Guide
===========

.. image:: _static/Overview.png
    :width: 100 %

This user guide provides an overview of the important elements in a TOPFARM
optimization. To look into details of available options or function inputs/outputs,
please see the :ref:`api`. Examples of TOPFARM code can be found in the
:ref:`Examples` section.


TopFarm Problem
----------------

    The ``TopFarmProblem`` object is the top-level object for performing optimizations.
    It is analogous to OpenMDAO's ``Problem`` when constructing workflows. Defining a
    TopFarmProblem involves specifying the following:  
    
    * Design variables  
    * Cost component  
    * Optimization driver (optional)  
    * Constraints (optional)  
    * Plotting component (optional)  

    Once initialized, a TopFarmProblem will have a method called ``optimize`` that runs
    the optimization. The optimization can return three items: ``cost``, ``state``, and
    ``recorder``. The ``cost`` object is the final result of the optimization. The ``state``
    provides the final values of the design variables for the optimization. Lastly,
    the ``recorder`` presents a record of which sets of design variables were tried
    during the optimization.


Design Variables
-----------------

    These are the variables that should be changed during the optimization. Common
    options include wind turbine positions and/or turbine types.
	

Cost Component
----------------

    At its most simplest, the cost component is the object that calculates the
    objective function for the optimization (e.g., AEP). For nested optimizations,
    this cost component could actually be another TopFarmProblem. The cost component
    must be in the style of an OpenMDAO v2 ExplicitComponent, but pure-Python cost
    functions can be wrapped using the ``CostModelComponent`` class in 
    ``topfarm.cost_models.cost_model_wrappers``. TOPFARM also contains a wrapper
    for PyWake's AEP calculator, which can be used with a variety of wake models.

	
DTU Cost Model
----------------	

	This is the DTU offshore cost model. The class includes methods for simple calculations
	of the Internal Rate of Return (IRR) and Net Present Value (NPV). Further, it breaks up
	the project costs into DEVEX, CAPEX, OPEX and ABEX within seperate methods, which may be
	called individually. It generally relies on curve fitting using input parameters such as 
	rated power or water depth, and was tuned using data obtained from the 
	industry. It supports three types of drivetrains: high-speed, medium-speed and direct-drive.
	It also supports two types of offshore foundations: monopile and jacket. The model was originally created in Excel by Kenneth Thomse and
	Søren Oemann Lind and Rasmus (publication forthcoming). The monetary units are 2017 Euro. 

	
Drivers
----------

    The optimization driver is what is used to perform the numerical optimization.
    TOPFARM includes several "easy drivers" that work out-of-the-box, but custom
    optimization drivers can be coupled into a workflow. The default driver is the
    EasyScipyOptimize driver, which by default will use the SLSQP algorithm built
    into ``scipy``.


Constraints
-------------

    The constraints are the limitations that are set upon the design variables.
    The most common constraints are spacing constraints between turbines or
    boundaries of the wind farm. The provided constraints must be a list of the
    constraint components for the optimization. More details on the available
    constraint components in TOPFARM can be found in the :ref:`api`. There are
    no constraints by default.


Plotting Component
-------------------

    This component allows the user to see the state of the wind farm as it is
    being optimized. The default option is to not plot during optimization.
