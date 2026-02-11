.. TOPFARM documentation master file.
Welcome to TOPFARM
===========================================
.. container:: topfarm-logo-frontpage

   .. image:: _static/TopFarm_logo.svg
      :width: 60%
      :align: center

TOPFARM is a Python package developed by *DTU Wind Energy* that serves as a wind farm optimizer for both onshore and offshore wind farms. It uses the `OpenMDAO <http://openmdao.org/>`_ package for optimization and wraps the `PyWake <https://topfarm.pages.windenergy.dtu.dk/PyWake/>`_ package for easy computation of a wind farm’s Annual Energy Production (AEP).

In addition, it can compute other metrics such as the Internal Rate of Return (IRR) and Net Present Value (NPV) and utilizes different engineering wake models available in PyWake to perform the flow simulations.

What types of problems can TOPFARM solve?
-------------------------------------------

.. image:: _static/topfarm.png
    :target: _static/topfarm.png
    :width: 80 %
    :align: center

Over the years, TOPFARM has become a highly versatile tool that is capable of solving several types of optimization problems with different design variables and objectives functions in mind. Throughout its development, TOPFARM has evolved from simple layout optimization problems to more complex and relevant wind farm optimization scenarios. Its capabilities and range were designed for both research and industry related topics. Today, TOPFARM can provide the user solutions in:

    * Wind farm layout optimization for different turbine types
    * Wind farm layout optimization for different turbine hub heights
    * Active control (wake steering) optimization
    * Load constrained layout optimization
    * Load constrained wake steering optimization
    * Optimization with bathymetry
    * LCoE-based layout optimization
	
Additionally, the objective function in TOPFARM can be formulated in economical terms, that is with the inclusion of several financial factors that are inherent in the wind farm design process. These can include the financial balance, foundation costs, electrical costs (cabling), fatigue degradation of turbine components and Operation and Management (O&M) costs.

The calculations for the wind farm interactions are done through PyWake, which is responsible for computing the wake losses and power production of both individual turbines and whole wind farms with the use of `engineering wake models <https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html>`_. In TOPFARM, the objective function is evaluated by the cost model component, and can be represented by either power production or financial goals.

TOPFARM comes with many built-in wake and cost models that were designed to accurately represent the optimization problem at hand. However, the tool is very flexible, and users are also able to perform custom optimizations as well.

For installation instructions, please see the :ref:`Installation Guide <installation>`. The base code is open-source and freely available on `GitLab 
<https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2>`_ (MIT license).

Getting Started
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration of a TOPFARM problem can increase in complexity depending on the case study at hand. For the basic tool capabilities, please refer to the :ref:`basic example <basic_examples>` section. The more elaborated wind farm optimization examples are shown in the :ref:`advanced examples <advanced_examples>` section. For new users, the :ref:`User Guide <user_guide>` provides detailed information about the components within TOPFARM and their description.

Can I get a private/commercial version of TOPFARM?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For proprietary developers, we offer the option of having a short-term private
repository for co-development of cutting-edge plugins. Please contact the
`TOPFARM development team <mailto:mikf@dtu.dk>`_ for further details.


How can I contribute to TOPFARM?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We encourage contributions from different developers. You can contribute by
submitting an issue using TOPFARM's `Issue Tracker <https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2/issues>`_
or by volunteering to resolve an issue already in the queue.

Citation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you are using TOPFARM, please cite it using:
    Mads M. Pedersen, Mikkel Friis-Møller, Pierre-Elouan Réthoré, Ernestas Simutis, Riccardo Riva, Julian Quick, Nikolay Krasimirov Dimitrov, Jenni Rinker, & Katherine Dykes. (2025). DTUWindEnergy/TopFarm2: Release of v2.6.1 (v2.6.1). Zenodo. https://doi.org/10.5281/zenodo.17540961

or

.. code-block:: bibtex
   :linenos:

   @software{mads_m_pedersen_2025_17540961,
     author       = {Mads M. Pedersen and
                     Mikkel Friis-Møller and
                     Pierre-Elouan Réthoré and
                     Ernestas Simutis and
                     Riccardo Riva and
                     Julian Quick and
                     Nikolay Krasimirov Dimitrov and
                     Jenni Rinker and
                     Katherine Dykes},
     title        = {DTUWindEnergy/TopFarm2: Release of v2.6.1},
     month        = nov,
     year         = 2025,
     publisher    = {Zenodo},
     version      = {v2.6.1},
     doi          = {10.5281/zenodo.17540961},
     url          = {https://doi.org/10.5281/zenodo.17540961},
     swhid        = {swh:1:dir:0b90c25a28df70dd7739ff61e9077571697874c5
                     ;origin=https://doi.org/10.5281/zenodo.3247031;
                     visit=swh:1:snp:1871c10015766a303da42aff0cbb412d38ac9652;
                     anchor=swh:1:rel:6a7020bd9717aabad26630b0c1dafde08447fe5b;
                     path=DTUWindEnergy-TopFarm2-fc9cb8d},
   }


Package Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    .. toctree::
        :maxdepth: 2
	:caption: Contents

        installation
        user_guide
        basic_examples
        advanced_examples
        api

    .. toctree::
        :glob:
        :maxdepth: 0
	:caption: Publications
    
        pub_tool
        pub_theses
        pub_related
