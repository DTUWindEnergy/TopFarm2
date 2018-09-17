.. _using_topfarm:

===========================
Using TOPFARM
===========================

Once you have installed TOPFARM, you are ready to begin creating and running
your own optimization problems. You can use any Python development workflow
that you like or are accustomed to. For new users unfamiliar with Python, and
especially those on Windows, we recommend using either the Spyder IDE or
Jupyter notebooks. Spyder is an IDE that is very similar to the Matlab
interface. Jupyter notebooks are very similar to Mathematica.

Please note that TOPFARM is installed inside its own environment, which means
that you must do a few extra steps to get Spyder to use the correct kernel.
More details are provided below.

Spyder
-------------------

Spyder is a program that looks and behaves very similar to the Matlab
development environment. If you downloaded Anaconda, you already have spyder
in your base environment. If you downloaded Miniconda, you may need to install
it into your base environment::

   activate base
   conda install spyder -y

Note that the insructions below are only valid if your base environment has
Python 3.X. If your base environment has Python 2.X and not Python 3.X, you
will need to install a Python 3.X version of Spyder, either in its own Python 
3.X environment or in the ``pyTopfarm`` environment::

   activate <environment name>
   conda install spyder -y
   
To run TOPFARM from Spyder, you must configure Spyder to use the correct Python
kernel. By default, Spyder will use the base kernel, but TOPFARM is installed
in its own environment. To use the correct kernel from the TOPFARM environment,
do the following:

1. Click on "Tools" -> "Preferences" -> "Python interpreter".  
2. Click "Use the following interpreter", then navigate to the ``python.exe``
   that corresponds to the pyTopfarm environment. On Windows, this is probably
   in a place like 
   ``C:\users\<user>\AppData\Local\Continuum\Anaconda\envs\pyTopfarm\python.exe``.  
3. Try to open a new IPython console in Spyder. If you get an error that certain
   packages are missing, you will need to install them into your pyTopfarm
   environment by opening an Anaconda Prompt and installing them::

      activate pyTopfarm
      conda install <missing package(s)>

   Once you have installed the packages, try again to open an IPython console
   in Spyder. It should work now.

If you ever need to switch back to your standard kernel, follow these
instructions but click the "Default" radio button for the Python kernel.
   

Jupyter Notebooks
------------------

Jupyter notebooks are very user-friendly ways to present examples or run
scripts. They are more interactive than classic Python scripts, and the code
is contained in isolated cells, similar to Mathematica.

Unlike Spyder, Jupyter notebook can be run from the pyTopfarm environment
without any configuration changes. Simply open an Anaconda Prompt, ``cd``
into the directory with the notebook you want to run, then enter::

   jupyter notebook

Jupyter will launch your web browser, and from there you can click on the
notbook and run it.

**Important!!!** You should not close the notebook by closing your browser,
because this will not close the associated Python kernel. Instead, go to 
"File" -> "Close and halt". When you are done with your Jupyter session, 
you can close the browser and then hit Control+C on your Anaconda Prompt to
kill the Jupyter session.
