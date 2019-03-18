.. TOPFARM documentation master file.

Welcome to TOPFARM
===========================================
*- DTU Wind Energy's wind-farm optimizer*

What is TOPFARM? What problems can it solve?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TOPFARM is a Python package developed by DTU Wind Energy to help with wind-farm
optimizations. It uses the `OpenMDAO <http://openmdao.org/>`_
package for optimization and also wraps the
`PyWake <https://topfarm.pages.windenergy.dtu.dk/PyWake/>`_ package for easy
computation of annual energy production (AEP) with different wake models.

TOPFARM can solve many different types of problems: layout optimizations,
turbine-type optimizations, etc. Many of the most common wake and cost models
are built in, but users also have the option to couple their own wake or
cost model for custom optimizations.

How do I install it?
^^^^^^^^^^^^^^^^^^^^^^

The base code is open-source and freely available on `GitLab 
<https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2>`_ (MIT license).
The quick-start command to install the open-source code is::

    pip install topfarm

For more detailed installation instructions or options, please see the
:ref:`Installation Guide <installation>`.

Can I get a private/commercial version of TOPFARM?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For proprietary developers, we offer the option of having a short-term private
repository for co-development of cutting-edge plugins. Please contact the
`TOPFARM development team <mailto:dave@dtu.dk>`_ for further details.


How can I contribute to TOPFARM?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We encourage contributions from different developers. You can contribute by
submitting an issue using TOPFARM's `Issue Tracker <https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2/issues>`_
or by volunteering to resolve an issue already in the queue.


Package Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    .. toctree::
        :maxdepth: 2

        installation
        user_guide
        api
        examples
