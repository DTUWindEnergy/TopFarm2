.. _bookkeeping:


=======================
Bookkeeping
=======================


GitLab project settings
-----------------------


I think there's an artist hidden in the bottom of every single one of us.
This is the fun part.
This is your world.
Little trees and bushes grow however makes them happy.
Trees cover up a multitude of sins.
We'll paint one happy little tree right here.

Runners
-------

Isn't that fantastic that you can create an almighty tree that fast?
We might as well make some Almighty mountains today as well, what the heck.
Fluff it up a little and hypnotize it.
We're not trying to teach you a thing to copy.
We're just here to teach you a technique, then let you loose into the world.

Documentation generation
------------------------

1. Run ``sphinx-quickstart`` with the following options:  

  * root path: docs  
  * separate source and build: y  
  * Name prefix: '_'  
  * (Project name, Author, version, etc. particular to TOPFARM)  
  * Project language: en  
  * Source file suffix: .rst  
  * Name of master document: index  
  * Use the epub builder: n  
  * autodoc: y  
  * doctest: y  
  * intersphinx: y  
  * todo: y  
  * coverage: y  
  * imgmath: n  
  * mathjax: n  
  * ifconfig: n  
  * viewcode: y  
  * githubpages: n  
  * Makefile: y  
  * Windows command file: y  

2. Make a few changes to the generated ``conf.py``:  

  * Change ``html_theme`` to ``sphinx_rtd_theme``  
  * ``html_theme_options = {
    # Toc options
    'navigation_depth': 2,
    'collapse_navigation': False}``  
  * Remove ``html_sidebars``  
  * Add elements for intersphinx mapping (see ``conf.py``)  

3. Manually change ``index.rst`` for content that is not generated
   automatically.
