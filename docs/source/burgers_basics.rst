Burgers solvers
===============

Solvers for the inviscid and viscid Burgers' equation.


Inviscid
--------

The 2d inviscid Burgers' equations are given by:

.. math::

   u_t + u u_x + v u_y = 0,

   v_t + u v_x + v v_y = 0.

This is similar to the advection equation, however the speed is now both the quantity being advected and the speed at which it is moving.

The :py:mod:`burgers` solver implements the directionally unsplit corner transport upwind algorithm with piecewise linear reconstruction.

Viscid
------

The 2d viscid Burgers' equations are given by:

.. math::

   u_t + u u_x + v u_y = \nu \nabla^2 u,

   v_t + u v_x + v v_y = \nu \nabla^2 v,

where :math:`\nu` is the dynamic viscosity. The viscosity acts to smooth out shocks: rather than being infinitely thin, they will now have a finite width.

The viscid Burgers' equations are solved by the :py:mod:`burgers_viscid` solver. The advective parts of the equations are solved in the same was as for the inviscid case (though with the addition of a diffusive source term when we predict the interface states). The viscous source is solved implicitly using a Crank-Nicolson discretization (i.e. we solve it as a diffusion equation using the same method used by the :doc:`diffusion <diffusion_basics>` solver).


Parameters
----------

The main parameters that affect these solvers are:

+-------------------------------------------------------------------------------------------------------------------------------+
| ``[driver]``                                                                                                                  |
+=====================+=========================================================================================================+
|``cfl``              | the Burgers CFL number (what fraction of a zone can we cross in a single timestep)                      |
+---------------------+---------------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------------------------------------------------+
| ``[burgers]``                                                                                                                 |
+=====================+=========================================================================================================+
|``limiter``          | what type of limiting to use in reconstructing the slopes. 0 means use an unlimited second-order        |
|                     | centered difference. 1 is the MC limiter, and 2 is the 4th-order MC limiter                             |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``visc``             | the viscosity (`burgers_viscid` only)                                                                   |
+---------------------+---------------------------------------------------------------------------------------------------------+

Examples
--------

step
^^^^^^

The step problem initializes a step function, with the velocity in the x-direction initially 1 in the left half of the domain and 2 in the right half of the domain. As the simulation progresses, the intial discontinuity in the center of the domain spreads out to form a rarefaction. This is run as:

.. code-block:: none

   ./pyro.py burgers step inputs.step


sine
^^^^

The sine problem initializes the domain with sinusoidal data. Over time, this steepens to form a shock which propagates to the right. The problem is run as

.. code-block:: none

   ./pyro.py burgers sine inputs.sine


compare
^^^^^^^

This problem is designed for use with the `burgers_compare.py` script and the `burgers` solver. It initializes the domain with the data :math:`u(0, x) = x`. At time :math:`t`, the solution is :math:`u(t, x) = \frac{x}{1+t}`. The problem is run as

.. code-block:: none

   ./pyro.py burgers compare inputs.compare


Exercises
---------

The best way to learn these methods is to play with them yourself. The
exercises below are suggestions for explorations and features to add
to the Burgers solver.

Explorations
^^^^^^^^^^^^

* Test the convergence of the solver for a variety of initial
  conditions. Test with limiting on and off, and also test with the
  slopes set to 0 (this will reduce it down to a piecewise constant
  reconstruction method).

* Run without any limiting and look for oscillations and under and
  overshoots (does the advected quantity go negative in the tophat
  problem?)

Extensions
^^^^^^^^^^

* Implement a dimensionally split version of the Burgers
  algorithm. How does the solution compare between the unsplit and
  split versions? Look at the amount of overshoot and undershoot, for
  example.
