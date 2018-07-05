.. _installation:

===========================
Installation
===========================

There are three ways to use TOPFARM:

1. **Lightweight demo**
   (`jump to install <file:///C:/Users/rink/Documents/git/topfarm/TopFarm2/docs/build/html/user_guide/installation.html#lightweight-demo>`__)

   A small, light version of TOPFARM that is the easiest to install but is not
   appropriate for large problems. It utilizes a Docker image that comes with
   TOPFARM pre-installed in the image.

2. **Normal installation**
   (`jump to install <file:///C:/Users/rink/Documents/git/topfarm/TopFarm2/docs/build/html/user_guide/installation.html#normal-installation>`__)

   The standard version for non-developers running larger problems. The
   installation is done using Anaconda and requires a correctly configured
   gfortran compiler.

3. **Developer installation**
   (`jump to install <file:///C:/Users/rink/Documents/git/topfarm/TopFarm2/docs/build/html/user_guide/installation.html#developer-installation>`__)

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
`here <file:///C:/Users/rink/Documents/git/topfarm/TopFarm2/docs/build/html/user_guide/installation.html#configuring-windows-for-full-installation>`__).


Windows 7
^^^^^^^^^

1. Make sure your machine is correctly configured for a full installation (see below)
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
configured to work with Anaconda and a gfortran compiler (see instructions
`here <file:///C:/Users/rink/Documents/git/topfarm/TopFarm2/docs/build/html/user_guide/installation.html#configuring-windows-for-full-installation>`__).

These instructions are more terse because we expect developers to be slightly
more knowledgeable about tools such as ``git``, ``pip``, etc.

Windows 7
^^^^^^^^^

1. Make sure your machine is correctly configured for a full installation 
   (see below)
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


Configuring Windows for Full Installation
-----------------------------------------

In order for TOPFARM to work on a Windows machine, that machine should be
configured to have Anaconda with a correctly linked gfortran compiler.

The first step is to install `Anaconda <https://www.anaconda.com/download/>`_
for Python 3.6.

The second (and more complicated) step is to install a gfortran compiler
and configure it to work with Anaconda.

Windows 7
^^^^^^^^^

There are two options: an Intel Fortran compiler or the open-source MinGW-64.

* Install Intel Fortran compiler and activate it by entering the following
  command into a command prompt:
  ::

      "C:\\Program Files (x86)\\Intel\\Composer XE\\bin\\ifortvars.bat" intel64


* MinGW (instructions derived from `here <https://www.scivision.co/f2py-running-fortran-code-in-python-on-windows/>`__)

    1. Install MinGW-w64 from `Source Forge <https://sourceforge.net/projects/mingw-w64/>`_
       with the following options:
       
       * Architecture: ``x86_64``
       * Threads: ``posix``
       * Exception: ``seh``
       * Build revision: ``0``
       * Destination folder: ``c:\mingw``

    2. Add MinGW bin folder (``C:\mingw\mingw64\bin``) to your path variable
    3. Verify you can use gcc by typing ``gcc`` into a new Anaconda prompt and
       checking that there is a fatal error of ``no input files``
    4. If you do not have a file called ``distutils.cfg`` in one of the following
       locations, create it:

        1. ``c:\Anaconda\Lib\distutils\distutils.cfg`` **or**
        2. ``<user_folder>\AppData\Local\Continuum\Miniconda3\Lib\distutils\distutils.cfg``
    
    5. Add text to the config file or modify the existing file to have the
       following contents:
       ::

        [build]
        compiler=mingw32
