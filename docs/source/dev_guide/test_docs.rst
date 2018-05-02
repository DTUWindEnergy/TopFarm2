.. _test_docs:

===========================
Testing and Documentation
===========================

A fan brush is a fantastic piece of equipment.
Use it.
Make friends with it.
Go out on a limb - that's where the fruit is.



Coverage requirements
---------------------


I think there's an artist hidden in the bottom of every single one of us.
This is the fun part.
This is your world.
Little trees and bushes grow however makes them happy.
Trees cover up a multitude of sins.
We'll paint one happy little tree right here.


Docstring formatting
--------------------

Isn't that fantastic that you can create an almighty tree that fast?
We might as well make some Almighty mountains today as well, what the heck.
Fluff it up a little and hypnotize it.
We're not trying to teach you a thing to copy.
We're just here to teach you a technique, then let you loose into the world.

Updating the documentation
--------------------------

The workflow for updating the documentation is slightly different
than updating code. Namely, you must build the docs locally using
sphinx to make sure everything looks okay before you push. Here is
the workflow for updating documentation:

1. Create a branch ``docs`` from branch ``master``. If it already
   exists, contact whoever pushed last to make sure their changes
   were merged into ``master`` and have them delete the branch.
2. Make your change to the documentation.  
3. Locally build the documentation using whatever terminal prompt you
   like (e.g., Anaconda Prompt, etc) by executing this command from
   the ``docs`` folder on your computer:  
   ``make html``
4. In the created ``build/html`` folder, double-click ``index.html``
   and navigate around the resulting docs to make sure your changes
   are good.
5. Push your local ``docs`` branch to GitLab, create a merge request,
   and assign someone else to review it.  
6. If you're the reviewer, check the changes but also build the docs
   locally to see how they look.
7. Once the merge request is accepted, delete the ``docs`` branch
   both on GitLab and locally.
