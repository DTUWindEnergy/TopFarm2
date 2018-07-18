.. _installation:

===========================
Installation
===========================

There are three ways to use TOPFARM:

1. **Lightweight demo**
   (jump to install: `1. Lightweight Demo`_)

   A small, light version of TOPFARM that is the easiest to install but is not
   appropriate for large problems. It utilizes a Docker image that comes with
   TOPFARM pre-installed in the image.

2. **Normal installation**
   (jump to install: `2. Normal Installation`_)

   The standard version for non-developers running larger problems. The
   installation is done using Anaconda and requires a correctly configured
   gfortran compiler.

3. **Developer installation**
   (jump to install: `3. Developer Installation`_)

   For people that want to implement new functionalities or fix bugs. It
   follows a similar installation procedure as the normal installation, except
   that the files from both TOPFARM and its dependent libraries are installed
   locally using the editable flag in pip.


1. Lightweight Demo
-------------------

The first installation option is a lightweight demo option that can run simple
calculations but comes with a cost of small memory. This is a good option for
a quick installation and demonstration.

**Please note** this docker image does not exist yet.

Windows 7
^^^^^^^^^

1. Install `Docker Toolbox <https://docs.docker.com/toolbox/toolbox_install_windows/>`_.
2. Pull the docker image from docker hub
3. Activate a bash
4. Run the mini-example


2. Normal Installation
----------------------

This option will install TOPFARM using Anaconda. If you have a Windows
machine, then you must first ensure that your machine is configured to work
with Anaconda and a gfortran compiler (see instructions
`here <https://python-at-risoe.pages.windenergy.dtu.dk/compiling-on-windows/configuration.html>`__).


Windows 7
^^^^^^^^^

1. Make sure your machine is correctly configured to work with ``f2py`` and ``cython``.
   See intructions `here <https://python-at-risoe.pages.windenergy.dtu.dk/compiling-on-windows/configuration.html>`__.
2. Open your Anaconda prompt.
3. Change to the location where you want the TOPFARM files to be stored
   ::

     cd <path to folder>

4. Use git to clone the TOPFARM code
   ::

     git clone https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2.git

5. Change the prompt location to the new ``TopFarm2`` folder
   ::

     cd TopFarm2

6. Install TOPFARM and its dependencies using the ``install_topfarm.bat``
   file.
   ::

     install_topfarm.bat

7. Run the mini-example


3. Developer Installation
-------------------------

This option will install an editable version of TOPFARM using Anaconda. If you
have a Windows machine, then you must first ensure that your machine is
configured to work with Anaconda, a gfortran compiler, and a C compiler.
Detailed instructions can be found
`here <https://python-at-risoe.pages.windenergy.dtu.dk/compiling-on-windows/configuration.html>`__.

These instructions are more terse because we expect developers to be slightly
more knowledgeable about tools such as ``git``, ``pip``, etc.

Windows 7
^^^^^^^^^

1. Make sure your machine is correctly configured for a full installation. 
   See intructions `here <https://python-at-risoe.pages.windenergy.dtu.dk/compiling-on-windows/configuration.html>`__.
2. Create an environment for TOPFARM and activate it
3. Clone TOPFARM to your computer
4. Clone and install (using pip) the WindIO and FUSED-Wake libraries (URLs can
   be found in the ``install_topfarm.bat`` file)
   ::

     cd <path to local library>
     pip install -e .

5. Change back to your TOPFARM directory and install it with the editable flag
   enabled
   ::

      cd <path to TOPFARM files>
      pip install -e .

6. Run the mini-example
