.. _installation:

Installation
===========================


Pre-Installation
----------------------------

Before you can install the software, you must first have a working Python
distribution with a package manager. For all platforms we recommend that you
download and install Anaconda - a professional grade, full-blown scientific
Python distribution.

Install Anaconda, activate root environment:

    * Download and install Anaconda (Python 3.x version, 64 bit installer is recommended) from https://www.continuum.io/downloads
    
    * Update the root Anaconda environment (type in a terminal): 
        
        ``>> conda update --all``
    
    * Activate the Anaconda root environment in a terminal as follows: 
        
        ``>> activate``

If you have other Python programs besides TOPFARM, it is a good idea to install
each program in its own environment to ensure that the dependencies for the
different packages do not conflict with one another. The commands to create and
then activate an environment in an Anaconda prompt are::

   conda create --name topfarm python=3.6
   activate topfarm


Simple Installation
----------------------------

* Install from PyPi.org (official releases)::
  
    pip install topfarm

* Install from GitLab  (includes any recent updates)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2.git
        


Developer Installation
-------------------------------

We highly recommend developers install TOPFARM into its own environment. (See
instructions above.) The commands to clone and install TOPFARM with developer
options into the current active environment in an Anaconda Prommpt are as
follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2.git
   cd TopFarm2
   pip install -e .
