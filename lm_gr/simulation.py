"""
TODO: MAKE IT POSSIBLE TO START A PROGRAM FROM EXISTING OUTPUT FILE?
e.g. maybe do a dump of EVERYTHING at the end of the program so that it's possible to read this in and set up the problem in the state it left off.

TODO: D ln u0/Dt term in momentum equation?

FIXME: check edge/cell-centred/time-centred quantities used correctly

CHANGED: moved tov update in steps 4 and 8 to after the Dh0 update as it is a
function of Dh0

TODO: not entirely sure about the lateral averaging of D, Dh to get the base states in steps 4 and 8

CHANGED: made python3 compliable

TODO: nose testing!

Run with a profiler using
    python -m cProfile -o pyro.prof pyro.py
then view using
    snakeviz pyro.prof

All the keyword arguments of functions default to None as their default values
will be member variables which cannot be accessed in the function's argument
list.
"""

from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

from lm_gr.problems import *
import lm_gr.LM_gr_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
import multigrid.variable_coeff_MG as vcMG
from util import profile
import lm_gr.metric as metric


class Basestate(object):
    """
    Basestate is a class for the 1d base states. It has much the same indexing functionality as the 2d Grid2d class, as well as functions that allow easy conversion of the 1d arrays to 2d.
    """
    def __init__(self, ny, ng=0, d=None):
        self.ny = ny
        self.ng = ng
        self.qy = ny + 2 * ng

        if d is None:
            self.d = np.zeros((self.qy), dtype=np.float64)
        else:
            self.d = d

        self.jlo = ng
        self.jhi = ng + ny - 1

    def d2d(self):
        """
        Returns 2d version of entire 1d data array
        """
        return self.d[np.newaxis, :]

    def d2df(self, qx):
        """
        fortran compliable version
        FIXME: is there any benefit to using the above version? Otherwise, just scrap that and keep this one?
        """
        return np.array([self.d, ] * qx)

    def v(self, buf=0):
        """
        Data array without the ghost cells (i.e. just the interior cells)
        """
        return self.d[self.jlo-buf:self.jhi+1+buf]

    def v2d(self, buf=0):
        """
        2d version of the interior data.
        """
        return self.d[np.newaxis,self.jlo-buf:self.jhi+1+buf]

    def v2df(self, qx, buf=0):
        """
        fortran compliable version
        """
        return np.array(self.d[self.jlo-buf:self.jhi+1+buf, ] * qx)

    def v2dp(self, shift, buf=0):
        """
        2d version of the data array, shifted in the y direction and without ghost cells
        """
        return self.d[np.newaxis,self.jlo+shift-buf:self.jhi+1+shift+buf]

    def v2dpf(self, qx, shift, buf=0):
        """
        fortran compliable version
        """
        return np.array(self.d[self.jlo+shift-buf:self.jhi+1+shift+buf, ] * qx)

    def jp(self, shift, buf=0):
        """
        1d shifted without ghost cells
        """
        return self.d[self.jlo-buf+shift:self.jhi+1+buf+shift]

    def copy(self):
        """
        Return a deep (?) copy of the object.
        """
        return Basestate(self.ny, ng=self.ng, d=self.d.copy())

    def __add__(self, other):
        """
        Overrides standard addition
        """
        if isinstance(other, Basestate):
            return Basestate(self.ny, ng=self.ng, d=self.d + other.d)
        else:
            return Basestate(self.ny, ng=self.ng, d=self.d + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Basestate):
            return Basestate(self.ny, ng=self.ng, d=self.d - other.d)
        else:
            return Basestate(self.ny, ng=self.ng, d=self.d - other)

    def __mul__(self, other):
        if isinstance(other, Basestate):
            return Basestate(self.ny, ng=self.ng, d=self.d * other.d)
        else:
            return Basestate(self.ny, ng=self.ng, d=self.d * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Basestate):
            return Basestate(self.ny, ng=self.ng, d=self.d / other.d)
        else:
            return Basestate(self.ny, ng=self.ng, d=self.d / other)

    def __div__(self, other):
        if isinstance(other, Basestate):
            return Basestate(self.ny, ng=self.ng, d=self.d / other.d)
        else:
            return Basestate(self.ny, ng=self.ng, d=self.d / other)

    def __rdiv__(self, other):
        if isinstance(other, Basestate):
            return Basestate(self.ny, ng=self.ng, d=other.d / self.d)
        else:
            return Basestate(self.ny, ng=self.ng, d=other / self.d)

    def __rtruediv__(self, other):
        if isinstance(other, Basestate):
            return Basestate(self.ny, ng=self.ng, d=other.d / self.d)
        else:
            return Basestate(self.ny, ng=self.ng, d=other / self.d)


class Simulation(NullSimulation):

    def __init__(self, solver_name, problem_name, rp, timers=None, fortran=True):
        """
        Initialize the Simulation object

        Parameters
        ----------
        solver_name : str
            The name of the solver we wish to use. This should correspond
            to one of the solvers in the pyro folder.
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in lm_gr/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        fortran : boolean, optional
            Determines whether use the fortran smoother or the original
            python one.
        """

        NullSimulation.__init__(self, solver_name, problem_name, rp, timers=timers)

        self.base = {}
        self.aux_data = None
        self.metric = None
        self.dt_old = 1.
        self.fortran = fortran


    def initialize(self):
        """
        Initialize the grid and variables for low Mach general relativistic atmospheric flow
        and set the initial conditions for the chosen problem.
        """

        myg = grid_setup(self.rp, ng=4)

        bc_dens, bc_xodd, bc_yodd = bc_setup(self.rp)

        my_data = patch.CellCenterData2d(myg)

        my_data.register_var("density", bc_dens)
        my_data.register_var("enthalpy", bc_dens)
        my_data.register_var("x-velocity", bc_xodd)
        my_data.register_var("y-velocity", bc_yodd)

        # we'll keep the internal energy around just as a diagnostic
        my_data.register_var("eint", bc_dens)

        # phi -- used for the projections.  The boundary conditions
        # here depend on velocity.  At a wall or inflow, we already
        # have the velocity we want on the boundary, so we want
        # Neumann (dphi/dn = 0).  For outflow, we want Dirichlet (phi
        # = 0) -- this ensures that we do not introduce any tangental
        # acceleration.
        bcs = []
        # CHANGED: I think the neumann/dirichlet thing here was the wrong way around?
        for bc in [self.rp.get_param("mesh.xlboundary"),
                   self.rp.get_param("mesh.xrboundary"),
                   self.rp.get_param("mesh.ylboundary"),
                   self.rp.get_param("mesh.yrboundary")]:
            if bc == "periodic":
                bctype = "periodic"
            elif bc in ["reflect", "slipwall"]:
        #        bctype = "neumann"
                 bctype = "dirichlet"
            elif bc in ["outflow"]:
        #        bctype = "dirichlet"
                 bctype = "neumann"
            bcs.append(bctype)

        bc_phi = patch.BCObject(xlb=bcs[0], xrb=bcs[1], ylb=bcs[2], yrb=bcs[3])

        # CHANGED: tried setting phi BCs to same as density?
        #my_data.register_var("phi-MAC", bc_phi)
        #my_data.register_var("phi", bc_phi)
        my_data.register_var("phi-MAC", bc_dens)
        my_data.register_var("phi", bc_dens)

        # gradp -- used in the projection and interface states.  We'll do the
        # same BCs as density
        my_data.register_var("gradp_x", bc_dens)
        my_data.register_var("gradp_y", bc_dens)

        # passive scalar that is advected along - this is actually going to be the scalar times the density, so be careful when plotting at the end
        my_data.register_var("scalar", bc_dens)

        my_data.create()

        self.cc_data = my_data

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = patch.CellCenterData2d(myg)

        aux_data.register_var("coeff", bc_dens)
        aux_data.register_var("source_y", bc_yodd)
        aux_data.register_var("old_source_y", bc_yodd)
        aux_data.register_var("plot_me", bc_yodd) # somewhere to store data for plotting

        aux_data.create()
        self.aux_data = aux_data

        # we also need storage for the 1-d base state -- we'll store this
        # in the main class directly.
        self.base["D0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["Dh0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["p0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["old_p0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["U0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["U0_old_half"] = Basestate(myg.ny, ng=myg.ng)
        # U0(t=0) = 0 as an initial approximation

        # add metric
        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")

        alpha = Basestate(myg.ny, ng=myg.ng)

        # r = y + R, where r is measured from the centre of the star,
        # R is the star's radius and y is measured from the surface
        alpha.d[:] = np.sqrt(1. - 2. * g * (1. - myg.y[:]/R) / (R * c**2))

        beta = [0., 0.]

        gamma_matrix = np.zeros((myg.qx, myg.qy, 2, 2), dtype=np.float64)
        gamma_matrix[:, :, :,:] = 1. + 2. * g * \
            (1. - myg.y[np.newaxis, :, np.newaxis, np.newaxis] / R) / \
            (R * c**2) * np.eye(2)[np.newaxis, np.newaxis, :, :]

        self.metric = metric.Metric(self.cc_data, self.rp, alpha, beta,
                                    gamma_matrix)

        u0 = self.metric.calcu0()

        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.base, self.rp, self.metric)')

        # Construct zeta
        gamma = self.rp.get_param("eos.gamma")
        self.base["zeta"] = Basestate(myg.ny, ng=myg.ng)
        D0 = self.base["D0"]
        self.base["zeta"].d[:] = D0.d

        # we'll also need zeta on vertical edges -- on the domain edges,
        # just do piecewise constant
        self.base["zeta-edges"] = Basestate(myg.ny, ng=myg.ng)
        self.base["zeta-edges"].jp(1)[:] = \
            0.5 * (self.base["zeta"].v() + self.base["zeta"].jp(1))
        self.base["zeta-edges"].d[myg.jlo] = self.base["zeta"].d[myg.jlo]
        self.base["zeta-edges"].d[myg.jhi+1] = self.base["zeta"].d[myg.jhi]

        # initialise source
        S = self.aux_data.get_var("source_y")
        S = self.compute_S()
        oldS = self.aux_data.get_var("old_source_y")
        oldS = myg.scratch_array(data=S.d)
        #oldS.d[:,:] = S.d


    # This is basically unused now.
    @staticmethod
    def make_prime(a, a0):
        """
        Returns the perturbation to the base state.

        Parameters
        ----------
        a : Grid2d
            2d full state
        a0 : Basestate
            Corresponing 1d base state

        Returns
        -------
        make_prime : Grid2d
            2d perturbation, :math: `a-a_0`
        """
        return a - a0.v2d(buf=a0.ng)

    @staticmethod
    def smooth_neighbours(a, mask):
        """
        Returns smoothed version of state: points satisfying logical input will be replaced by gaussian smoothing of their immediate neighbours.

        Parameters
        ----------
        a : Grid2d
            2d full state
        mask : logical
            logical mask of which points to smooth

        Returns
        -------
        smooth_neighbours : float array
        """

        a.v()[mask] = ((a.jp(1)[mask] + a.jp(-1)[mask] + a.ip(1)[mask] +
                        a.ip(-1)[mask]) * 26. + (a.ip_jp(1,1)[mask] +
                        a.ip_jp(-1,1)[mask] + a.ip_jp(1,-1)[mask] +
                        a.ip_jp(-1,-1)[mask]) * 16. + a.v()[mask]) / 209.


    @staticmethod
    def lateral_average(a):
        r"""
        Calculates and returns the lateral average of a, assuming that stuff is
        to be averaged in the :math: `x` direction.

        Parameters
        ----------
        a : float array
            2d array to be laterally averaged

        Returns
        -------
        lateral_average : float array
            lateral average of a
        """
        return np.mean(a, axis=0, dtype=np.float64)


    def update_zeta(self, D0=None, zeta=None, u=None, v=None, u0=None):
        """
        Updates zeta in the interior and on the edges.
        Assumes all other variables are up to date.

        Parameters
        ----------
        D0 : Basestate object, optional
            base state (relativistic) density
        zeta : Basestate object, optional
            zeta
        u : ArrayIndexer object, optional
        """

        myg = self.cc_data.grid
        if D0 is None:
            D0 = self.base["D0"]
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        if zeta is None:
            zeta = self.base["zeta"]
        zeta_edges = self.base["zeta-edges"]

        try:
            zeta.d[:] = D0.d
        except FloatingPointError:
            print('D0: ', np.max(D0.d))
            print('u0: ', np.max(u0.d))

        # calculate edges
        zeta_edges.jp(1)[:] = 0.5 * (zeta.v() + zeta.jp(1))
        zeta_edges.d[myg.jlo] = zeta.d[myg.jlo]
        zeta_edges.d[myg.jhi+1] = zeta.d[myg.jhi]


    def compute_S(self, u=None, v=None):
        r"""
        :math: `S = -\Gamma^\mu_(\mu \nu) U^\nu `  (see eq 6.34, 6.37 in LowMachGR).
        base["source-y"] is not updated here as it's sometimes necessary to
        calculate projections of `S` (e.g. `S^{n*}`) and not `S^n`
        """
        myg = self.cc_data.grid
        S = myg.scratch_array()
        if u is None:
            u = self.cc_data.get_var("x-velocity")
        if v is None:
            v = self.cc_data.get_var("y-velocity")

        #chrls = np.array([[self.metric.christoffels([self.cc_data.t, i, j])
        #                   for j in range(myg.qy)] for i in range(myg.qx)])
        # time-independent metric
        chrls = self.metric.chrls

        S.d[:,:] = -(chrls[:,:,0,0,0] + chrls[:,:,1,1,0] + chrls[:,:,2,2,0] +
            (chrls[:,:,0,0,1] + chrls[:,:,1,1,1] + chrls[:,:,2,2,1]) * u.d +
            (chrls[:,:,0,0,2] + chrls[:,:,1,1,2] + chrls[:,:,2,2,2]) * v.d)

        return S


    def constraint_source(self, u=None, v=None, S=None, zeta=None):
        r"""
        calculate the source terms in the constraint, :math: `\zeta(S - \frac{dp}{dt}/ \Gamma_1 p)`

        Returns
        -------
        constraint : float array
            :math: `\zeta(S - \frac{dp}{dt}/ \Gamma_1 p)`
        """
        myg = self.cc_data.grid
        # get parameters
        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")
        gamma = self.rp.get_param("eos.gamma")
        #if u is None:
        #    u = self.cc_data.get_var("x-velocity")
        #if v is None:
        #    v = self.cc_data.get_var("y-velocity")
        if zeta is None:
            zeta = self.base["zeta"]
        if S is None:
            S = self.aux_data.get_var("source_y")

        p0 = self.base["p0"]
        dp0dt = Basestate(myg.ny, ng=myg.ng)
        # calculate dp0dt
        # FIXME: assumed it's 0 for now

        constraint = myg.scratch_array()
        # assumed adiabatic EoS so that Gamma_1 = gamma
        constraint.d[:,:] = zeta.d2df(myg.qx) * \
            (S.d - dp0dt.d2df(myg.qx) / (gamma * p0.d2df(myg.qx)))

        return constraint


    def calc_mom_source(self, u=None, v=None, Dh=None, Dh0=None, u0=None):
        r"""
        calculate the source terms in the momentum equation.
        This works only for the metric :math: `ds^2 = -a^2 dt^2 + 1/a^2 (dx^2 + dr^2)`

        FIXME: need the D_t ln u0 term?

        CHANGED: lowered first index of christoffels

        TODO: make this more general. This definitely needs to be done with
        einsum or something rather than by hand

        Returns
        -------
        mom_source :
            :math:`\Gamma_{\rho \nu j} U^\nu U^\rho -
                    \frac{\partial_j p_0}{Dh u_0}`
        """
        myg = self.cc_data.grid
        if u is None:
            u = self.cc_data.get_var("x-velocity")
        if v is None:
            v = self.cc_data.get_var("y-velocity")
        if Dh is None:
            Dh = self.cc_data.get_var("enthalpy")
        if Dh0 is None:
            Dh0 = self.base["Dh0"]
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        mom_source_r = myg.scratch_array()
        mom_source_x = myg.scratch_array()
        gtt = -(self.metric.alpha.d)**2
        gxx = 1. / self.metric.alpha.d**2
        grr = gxx
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v, u0=u0)

        #chrls = np.array([[self.metric.christoffels([self.cc_data.t, i, j])
        #                   for j in range(myg.qy)] for i in range(myg.qx)])
        # time-independent metric
        chrls = self.metric.chrls

        # note metric components needed to lower the christoffel symbols
        mom_source_x.d[:,:] = (gtt[np.newaxis,:] * chrls[:,:,0,0,1] +
            (gxx[np.newaxis,:] * chrls[:,:,1,0,1] +
             gtt[np.newaxis,:] * chrls[:,:,0,1,1]) * u.d +
            (grr[np.newaxis,:] * chrls[:,:,2,0,1] +
             gtt[np.newaxis,:] * chrls[:,:,0,2,1]) * v.d +
            gxx[np.newaxis,:] * chrls[:,:,1,1,1] * u.d**2 +
            grr[np.newaxis,:] * chrls[:,:,2,2,1] * v.d**2 +
            (grr[np.newaxis,:] * chrls[:,:,2,1,1] +
             gxx[np.newaxis,:] * chrls[:,:,1,2,1]) * u.d * v.d)
        mom_source_r.d[:,:] = (gtt[np.newaxis,:] * chrls[:,:,0,0,2] +
            (gxx[np.newaxis,:] * chrls[:,:,1,0,2] +
             gtt[np.newaxis,:] * chrls[:,:,0,1,2]) * u.d +
            (grr[np.newaxis,:] * chrls[:,:,2,0,2] +
             gtt[np.newaxis,:] * chrls[:,:,0,2,2]) * v.d +
            gxx[np.newaxis,:] * chrls[:,:,1,1,2] * u.d**2 +
            grr[np.newaxis,:] * chrls[:,:,2,2,2] * v.d**2 +
            (grr[np.newaxis,:] * chrls[:,:,2,1,2] +
             gxx[np.newaxis,:] * chrls[:,:,1,2,2]) * u.d * v.d)

        mom_source_r.d[:,:] -=  drp0.d[np.newaxis,:] / (Dh.d[:,:]*u0.d[:,:])

        mom_source_x.d[:,:] *=  self.metric.alpha.d2d()**2
        mom_source_r.d[:,:] *=  self.metric.alpha.d2d()**2

        return mom_source_x, mom_source_r


    def calc_psi(self, S=None, U0=None, p0=None, old_p0=None):
        r"""
        Calculate :math: `\psi`
        .. math::

            \psi = \partial_tp_0 + U_0\partial_r p_0

        Returns
        -------
        :math: `\psi`
        """
        myg = self.cc_data.grid
        if S is None:
            S = self.aux_data.get_var("source_y")
        if U0 is None:
            U0 = self.base["U0"]
        gamma = self.rp.get_param("eos.gamma")
        if p0 is None:
            p0 = self.base["p0"]
        if old_p0 is None:
            old_p0 = self.base["old_p0"]

        psi = Basestate(myg.ny, ng=myg.ng)

        psi.v(buf=myg.ng-1)[:] = gamma * 0.5 * \
            (p0.v(buf=myg.ng-1) + old_p0.v(buf=myg.ng-1)) * \
            (self.lateral_average(S.v(buf=myg.ng-1)) -
             (U0.jp(1, buf=myg.ng-1) - U0.v(buf=myg.ng-1)))

        return psi


    def base_state_forcing(self, U0_half=None, U0_old_half=None, Dh0_old=None, Dh0=None, u=None, v=None, u0=None):
        r"""
        calculate the base state velocity forcing term from 2C
        This works only for the metric :math: `ds^2 = -a^2 dt^2 + 1/a^2 (dx^2 + dr^2)`
        """
        myg = self.cc_data.grid
        drpi = myg.scratch_array()
        if U0_half is None:
            U0_half = self.base["U0"]
        if U0_old_half is None:
            U0_old_half = self.base["U0"]
        if Dh0 is None:
            Dh0 = self.base["Dh0"]
        if Dh0_old is None:
            Dh0_old = self.base["Dh0"]
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        Dh0_half = 0.5 * (Dh0_old + Dh0)

        drp0 = self.drp0(Dh0=Dh0_half, u=u, v=v, u0=u0)
        grr = 1. / self.metric.alpha.d**2

        #chrls = np.array([[self.metric.christoffels([self.cc_data.t, i, j])
        #                   for j in range(myg.qy)] for i in range(myg.qx)])
        # time-independent metric
        chrls = self.metric.chrls

        U0_star = Basestate(myg.ny, ng=myg.ng)
        U0_star.d[:] = (self.dt * U0_old_half.d +
                        self.dt_old * U0_half.d) / (self.dt + self.dt_old)

        drU0_old_half = Basestate(myg.ny, ng=myg.ng)
        drU0_half = Basestate(myg.ny, ng=myg.ng)
        drU0_star = Basestate(myg.ny, ng=myg.ng)
        drU0_old_half.d[1:-1] = (U0_old_half.d[2:] -
                                 U0_old_half.d[:-2]) / (2. * myg.dy)
        drU0_half.d[1:-1] = (U0_half.d[2:] -
                                 U0_half.d[:-2]) / (2. * myg.dy)
        drU0_star.d[:] = (self.dt * drU0_half.d +
                    self.dt_old * drU0_half.d) / (self.dt + self.dt_old)

        drp0_old_half = Basestate(myg.ny, ng=myg.ng)
        drp0_half = Basestate(myg.ny, ng=myg.ng)
        drp0_star = Basestate(myg.ny, ng=myg.ng)
        # don't have an old p0 atm - just ignore for now
        drp0_star.d[:] = drp0.d

        drpi.d[:,:] = \
            -0.5*(U0_half.d2d() - U0_old_half.d2d())/(self.dt + self.dt_old) -\
            U0_star.d2d() * drU0_star.d2d() - drp0_star.d2d() / (Dh0_half.d * u0.d) + \
            grr * chrls[:,:,2,2,0] * U0_star.d2d()**2

        return drpi


    def react_state(self, S=None, D=None, Dh=None, scalar=None, Dh0=None, u=None, v=None, u0=None):
        """
        gravitational source terms in the continuity equation (called react
        state to mirror MAESTRO as here they just have source terms from the
        reactions)
        """
        myg = self.cc_data.grid

        if D is None:
            D = self.cc_data.get_var("density")
        if Dh is None:
            Dh = self.cc_data.get_var("enthalpy")
        if scalar is None:
            scalar = self.cc_data.get_var("scalar")
        if v is None:
            v = self.cc_data.get_var("y-velocity")
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v, u0=u0)
        if S is None:
            S = self.aux_data.get_var("source_y")

        Dh.d[:,:] += 0.5 * self.dt * (S.d * Dh.d + u0.d * v.d * drp0.d2d())

        D.d[:,:] += 0.5 * self.dt * (S.d * D.d)

        scalar.d[:,:] += 0.5 * self.dt * (S.d * scalar.d)


    def advect_base_density(self, D0=None, U0=None):
        r"""
        Updates the base state density through one timestep. Eq. 6.131:
        .. math::

            D_{0j}^\text{out} = D_{0j}^\text{in} - \frac{\Delta t}{\Delta r}\left[\left(D_0^{\text{out},n+\sfrac{1}{2}} U_0^\text{in}\right)_{j+\sfrac{1}{2}} - \left(D_0^{\text{out},n+\sfrac{1}{2}} U_0^\text{in}\right)_{j-\sfrac{1}{2}}\right]

        This is incorrect as need to use edge based :math: `D`, as found in the
        evolve funciton.
        """
        myg = self.cc_data.grid
        if D0 is None:
            D0 = self.base["D0"]
        dt = self.dt
        dr = myg.dy
        if U0 is None:
            U0 = self.base["U0"]
        # FIXME: time-centred edge states

        D0.v()[:] += -(D0.jp(1) * U0.jp(1) - D0.v() * U0.v()) * dt / dr

    def advect_scalar(self):
        """
        Advects the scalar very naively.
        """
        myg = self.cc_data.grid
        scalar = self.cc_data.get_var("scalar")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        dt = self.dt
        dx = myg.dx
        dr = myg.dy

        scalar.v()[:,:] += \
            -(scalar.ip(1)*u.ip(1) - scalar.ip(-1)*u.ip(-1))*0.5*dt / dx \
            -(scalar.jp(1)*v.jp(1) - scalar.jp(-1)*v.jp(-1))*0.5*dt / dr

        self.cc_data.fill_BC("scalar")


    def enforce_tov(self, p0=None, Dh0=None, u=None, v=None, u0=None):
        r"""
        enforces the TOV equation. This is the GR equivalent of enforce_hse.
        Eq. 6.133.
        .. math::

            p_{0,j+1}^\text{out} = p_{0,j}^\text{in} + \frac{\Delta r}{2} \left(\partial_r p_{0,j+1} + \partial_r p_{0,j} \right)
        """
        myg = self.cc_data.grid
        if p0 is None:
            p0 = self.base["p0"]
        old_p0 = self.base["old_p0"]
        old_p0 = Basestate(myg.ny, ng=myg.ng)
        old_p0.d[:] = p0.d
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v, u0=u0)

        p0.d[1:] = p0.d[:-1] + 0.5 * self.cc_data.grid.dy * \
                   (drp0.d[1:] + drp0.d[:-1])


    def drp0(self, Dh0=None, u=None, v=None, u0=None):
        """
        Calculate drp0 as it's messy using eq 6.136
        """
        myg = self.cc_data.grid
        if Dh0 is None:
            Dh0 = self.base["Dh0"]
        # TODO: maybe instead of averaging u0, should calculate it
        # based on U0?
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        u01d = Basestate(myg.ny, ng=myg.ng)
        u01d.d[:] = self.lateral_average(u0.d)
        alpha = self.metric.alpha
        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")

        drp0 = Basestate(myg.ny, ng=myg.ng)

        drp0.d[:] = -Dh0.d * g / (R * c**2 * alpha.d**2 * u01d.d)

        return drp0


    def advect_base_enthalpy(self, Dh0=None, S=None, U0=None, u=None, v=None, u0=None):
        r"""
        updates base state enthalpy throung one timestep using eq. 6.134

        .. math::

            (Dh)_{0j}^\text{out}
                = (Dh)_{0j}^\text{in} - \frac{\Delta t}{\Delta r}\left(\left[(Dh)_0^{n+\sfrac{1}{2}} U_0^\text{in}\right]_{j+\sfrac{1}{2}} - \left[(Dh)_0^{n+\sfrac{1}{2}} U_0^\text{in}\right]_{j-\sfrac{1}{2}}\right) + \Delta tu^0\psi_j^\text{in}
        """
        myg = self.cc_data.grid
        if Dh0 is None:
            Dh0 = self.base["Dh0"]
        dt = self.dt
        dr = myg.dy
        if U0 is None:
            U0 = self.base["U0"]
        if S is None:
            S = self.aux_data.get_var("source_y")
        psi = self.calc_psi(U0=U0, S=S)
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)

        # FIXME: find out how to find the time-centred edge
        # states and use them here?
        # CHANGED: add psi

        Dh0.v()[:] += -(Dh0.jp(1) * U0.jp(1) - Dh0.v() * U0.v()) * dt / dr + \
                      dt * self.lateral_average(u0.v()) * psi.v()
                      # dt * U0.v() * self.drp0().v() + \


    def compute_base_velocity(self, U0=None, p0=None, S=None, Dh0=None, u=None, v=None, u0=None):
        r"""
        Calculates the base velocity using eq. 6.138

        .. math::

            \frac{ U^\text{out}_{0,j+\sfrac{1}{2}} -  U^\text{out}_{0,j-\sfrac{1}{2}}}{\Delta r} = \left(\bar{S}^\text{in} - \frac{1}{\bar{\Gamma}_1^\text{in} p_0^\text{in}}\psi^\text{in}\right)_j
        """
        myg = self.cc_data.grid
        if p0 is None:
            p0 = self.base["p0"]
        dt = self.dt
        dr = myg.dy
        if U0 is None:
            U0 = self.base["U0"]
        gamma = self.rp.get_param("eos.gamma")
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v, u0=u0)
        if S is None:
            S = self.aux_data.get_var("source_y")

        # Sbar = latavg(S)
        Sbar = self.lateral_average(S.d)
        U0.d[0] = 0.
        # FIXME: fix cell-centred / edge-centred indexing.
        U0.d[1:] = U0.d[:-1] + dr * (Sbar[:-1] - U0.d[:-1] * drp0.d[:-1] /
                                     (gamma * p0.d[:-1]))


    def compute_timestep(self, u0=None):
        """
        The timestep() function computes the advective timestep
        (CFL) constraint.  The CFL constraint says that information
        cannot propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        self.dt_old = self.dt

        myg = self.cc_data.grid

        cfl = self.rp.get_param("driver.cfl")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # the timestep is min(dx/|u|, dy|v|)
        xtmp = ytmp = 1.e33
        if not abs(u).max() < 1.e-25:
            xtmp = myg.dx / abs(u.v()).max()
        if not abs(v).max() < 1.e-25:
            ytmp = myg.dy / abs(v.v()).max()

        dt = cfl * min(xtmp, ytmp)

        # We need an alternate timestep that accounts for buoyancy, to
        # handle the case where the velocity is initially zero.
        Dh0 = self.base["Dh0"]
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)

        drp0 = self.drp0(Dh0=Dh0, u=u, v=v, u0=u0)

        _, mom_source_r = self.calc_mom_source(u0=u0)

        F_buoy = np.max(np.abs(mom_source_r.d))

        dt_buoy = np.sqrt(2.0 * myg.dx / max(F_buoy, 1.e-12))
        self.dt = min(dt, dt_buoy)
        if self.dt > 1.e30:
            self.dt = 100.
        if self.verbose > 0:
            print("timestep is {}".format(self.dt))


    def preevolve(self):
        """
        preevolve is called before we being the timestepping loop.  For
        the low Mach solver, this does an initial projection on the
        velocity field and then goes through the full evolution to get the
        value of phi.  The fluid state (D, u, v) is then reset to values
        before this evolve.
        """

        self.in_preevolve = True

        myg = self.cc_data.grid

        D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        scalar = self.cc_data.get_var("scalar")

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")
        self.cc_data.fill_BC("scalar")

        oldS = self.aux_data.get_var("old_source_y")
        oldS.d[:,:] = self.aux_data.get_var("source_y").d

        # a,b. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        self.react_state()

        # the coefficent for the elliptic equation is zeta^2/Dh u0
        u0 = self.metric.calcu0()
        coeff = 1. / (Dh * u0)
        zeta = self.base["zeta"]
        try:
            coeff.v()[:,:] *= zeta.v2d()**2
        except FloatingPointError:
            print('zeta: ', np.max(zeta.d))

        # next create the multigrid object.  We defined phi with
        # the right BCs previously
        mg = vcMG.VarCoeffCCMG2d(myg.nx, myg.ny,
                                 xl_BC_type=self.cc_data.BCs["phi"].xlb,
                                 xr_BC_type=self.cc_data.BCs["phi"].xrb,
                                 yl_BC_type=self.cc_data.BCs["phi"].ylb,
                                 yr_BC_type=self.cc_data.BCs["phi"].yrb,
                                 xmin=myg.xmin, xmax=myg.xmax,
                                 ymin=myg.ymin, ymax=myg.ymax,
                                 coeffs=coeff,
                                 coeffs_bc=self.cc_data.BCs["density"],
                                 verbose=0)

        # first compute div{zeta U}
        div_zeta_U = mg.soln_grid.scratch_array()

        # u/v are cell-centered, divU is cell-centered
        div_zeta_U.v()[:,:] = \
            0.5 * zeta.v2df(myg.qx) * (u.ip(1) - u.ip(-1)) / myg.dx + \
            0.5 * (zeta.v2dpf(myg.qx, 1) * v.jp(1) - \
            zeta.v2dpf(myg.qx, -1) * v.jp(-1)) / myg.dy

        # solve D (zeta^2/Dh u0) G (phi/zeta) = D( zeta U )
        constraint = self.constraint_source()
        # set the RHS to divU and solve
        mg.init_RHS(div_zeta_U.v(buf=1) - constraint.v(buf=1))
        mg.solve(rtol=1.e-12, fortran=self.fortran)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.cc_data.get_var("phi")
        phi.d[:,:] = mg.get_solution(grid=myg).d

        # get the cell-centered gradient of phi and update the
        # velocities
        # FIXME: this update only needs to be done on the interior
        # cells -- not ghost cells
        gradp_x, gradp_y = mg.get_solution_gradient(grid=myg)
        #pdb.set_trace()

        coeff = 1. / (Dh * u0)
        coeff.v()[:,:] *= zeta.v2d()

        u.v()[:,:] -= coeff.v() * gradp_x.v()
        v.v()[:,:] -= coeff.v() * gradp_y.v()

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        # c. now get an approximation to gradp at n-1/2 by going through the
        # evolution.

        # store the current solution -- we'll restore it in a bit
        orig_data = patch.cell_center_data_clone(self.cc_data)

        # get the timestep
        self.compute_timestep(u0=u0)

        # evolve
        self.evolve()

        # update gradp_x and gradp_y in our main data object
        new_gp_x = self.cc_data.get_var("gradp_x")
        new_gp_y = self.cc_data.get_var("gradp_y")

        orig_gp_x = orig_data.get_var("gradp_x")
        orig_gp_y = orig_data.get_var("gradp_y")

        orig_gp_x.d[:,:] = new_gp_x.d[:,:]
        orig_gp_y.d[:,:] = new_gp_y.d[:,:]

        self.cc_data = orig_data

        if self.verbose > 0:
            print("done with the pre-evolution")

        self.in_preevolve = False


    def evolve(self):
        """
        Evolve the low Mach system through one timestep.
        """

        D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        gradp_x = self.cc_data.get_var("gradp_x")
        gradp_y = self.cc_data.get_var("gradp_y")
        scalar = self.cc_data.get_var("scalar")

        u0 = self.metric.calcu0()

        # note: the base state quantities do not have valid ghost cells
        self.update_zeta(u0=u0)
        zeta = self.base["zeta"]
        zeta_edges = self.base["zeta-edges"]

        D0 = self.base["D0"]
        Dh0 = self.base["Dh0"]
        U0 = self.base["U0"]

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid

        oldS = self.aux_data.get_var("old_source_y")
        plot_me = self.aux_data.get_var("plot_me")

        #---------------------------------------------------------------------
        # create the limited slopes of D, Dh, u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("lm-gr.limiter")
        if limiter == 0:
            limitFunc = reconstruction_f.nolimit
        elif limiter == 1:
            limitFunc = reconstruction_f.limit2
        else:
            limitFunc = reconstruction_f.limit4

        #ldelta_rx = limitFunc(1, D.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0x = limitFunc(1, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        #ldelta_ex = limitFunc(1, Dh.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0x = limitFunc(1, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_ux = limitFunc(1, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v.d, myg.qx, myg.qy, myg.ng)

        #ldelta_ry = limitFunc(2, D.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0y = limitFunc(2, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        #ldelta_ey = limitFunc(2, Dh.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_uy = limitFunc(2, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v.d, myg.qx, myg.qy, myg.ng)

        #---------------------------------------------------------------------
        # 1. React state through dt/2
        #---------------------------------------------------------------------
        D_1 = myg.scratch_array(data=D.d)
        #D_1.d[:,:] = D.d
        Dh_1 = myg.scratch_array(data=Dh.d)
        #Dh_1.d[:,:] = Dh.d
        scalar_1 = myg.scratch_array(data=scalar.d)
        #scalar_1.d[:,:] = scalar.d
        self.react_state(D=D_1, Dh=Dh_1, scalar=scalar_1, u0=u0)

        #---------------------------------------------------------------------
        # 2. Compute provisional S, U0 and base state forcing
        #---------------------------------------------------------------------
        S = self.aux_data.get_var("source_y")
        S_t_centred = myg.scratch_array()

        if self.cc_data.t == 0:
            S_t_centred.d[:,:] = 0.5 * (oldS.d + S.d)
        else:
            S_t_centred.d[:,:] = S.d + \
                self.dt * 0.5 * (S.d - oldS.d) / self.dt_old

        U0_half_star = Basestate(myg.ny, ng=myg.ng)
        U0_half_star.d[:] = U0.d
        self.compute_base_velocity(U0=U0_half_star, S=S_t_centred, u0=u0)

        # FIXME: base state forcing? Where is it actually used??

        #---------------------------------------------------------------------
        # 3. get the advective velocities
        #---------------------------------------------------------------------

        """
        the advective velocities are the normal velocity through each cell
        interface, and are defined on the cell edges, in a MAC type
        staggered form

                         n+1/2
                        v
                         i,j+1/2
                    +------+------+
                    |             |
            n+1/2   |             |   n+1/2
           u        +     U       +  u
            i-1/2,j |      i,j    |   i+1/2,j
                    |             |
                    +------+------+
                         n+1/2
                        v
                         i,j-1/2

        """

        # this returns u on x-interfaces and v on y-interfaces.  These
        # constitute the MAC grid
        if self.verbose > 0:
            print("  making MAC velocities")

        # create the coefficient to the grad (pi/zeta) term
        # FIXME: Dh or Dh_1 here??
        coeff = self.aux_data.get_var("coeff")
        coeff.d[:,:] = 1.0 / (Dh.d * u0.d)
        # zeta here function of D0^n
        coeff.d[:,:] *= zeta.d2d()
        self.aux_data.fill_BC("coeff")

        g = self.rp.get_param("lm-gr.grav")

        mom_source_x, mom_source_r = self.calc_mom_source(Dh=Dh, u0=u0)

        _um, _vm = lm_interface_f.mac_vels(myg.qx, myg.qy, myg.ng,
                                           myg.dx, myg.dy, self.dt,
                                           u.d, v.d,
                                           ldelta_ux, ldelta_vx,
                                           ldelta_uy, ldelta_vy,
                                           coeff.d*gradp_x.d,
                                           coeff.d*gradp_y.d,
                                           mom_source_r.d)

        u_MAC = patch.ArrayIndexer(d=_um, grid=myg)
        v_MAC = patch.ArrayIndexer(d=_vm, grid=myg)
        # v_MAC is very small here but at least it's non-zero
        # entire thing sourced by Gamma^t_tr

        #---------------------------------------------------------------------
        # do a MAC projection to make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve D (beta_0^2/D) G phi = D (beta_0 U^MAC), where
        # phi is cell centered, and U^MAC is the MAC-type staggered
        # grid of the advective velocities.

        if self.verbose > 0:
            print("  MAC projection")

        # create the coefficient array: zeta**2/Dh u0
        # MZ!!!! probably don't need the buf here
        # use u0^n, so use U
        # Dh^n, not Dh^1
        coeff.v(buf=1)[:,:] = 1. / (Dh.v(buf=1) * u0.v(buf=1))
        # use zeta^n here, so use U
        coeff.v(buf=1)[:,:] *= zeta.v2d(buf=1)**2

        # create the multigrid object
        mg = vcMG.VarCoeffCCMG2d(myg.nx, myg.ny,
                                 xl_BC_type=self.cc_data.BCs["phi-MAC"].xlb,
                                 xr_BC_type=self.cc_data.BCs["phi-MAC"].xrb,
                                 yl_BC_type=self.cc_data.BCs["phi-MAC"].ylb,
                                 yr_BC_type=self.cc_data.BCs["phi-MAC"].yrb,
                                 xmin=myg.xmin, xmax=myg.xmax,
                                 ymin=myg.ymin, ymax=myg.ymax,
                                 coeffs=coeff,
                                 coeffs_bc=self.cc_data.BCs["density"],
                                 verbose=0)

        # first compute div{zeta U}
        div_zeta_U = mg.soln_grid.scratch_array()

        # MAC velocities are edge-centered.  div{zeta U} is cell-centered.
        div_zeta_U.v()[:,:] = \
            zeta.v2d() * (u_MAC.ip(1) - u_MAC.v()) / myg.dx + \
            (zeta_edges.v2dp(1) * v_MAC.jp(1) -
             zeta_edges.v2d() * v_MAC.v()) / myg.dy
        # careful: this makes u0_MAC edge-centred.
        u0_MAC = self.metric.calcu0(u=u_MAC, v=v_MAC)
        constraint = self.constraint_source(u=u_MAC, v=v_MAC, S=S_t_centred)

        # solve the Poisson problem
        mg.init_RHS(div_zeta_U.d - constraint.v(buf=1))
        mg.solve(rtol=1.e-12, fortran=self.fortran)

        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/zeta
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC.d[:,:] = mg.get_solution(grid=myg).d
        # this is zero and shouldn't be

        coeff = self.aux_data.get_var("coeff")
        # FIXME: is this u0 or u0_MAC?
        coeff.d[:,:] = self.metric.alpha.d2d()**2 / (Dh.d * u0.d)
        coeff.d[:,:] *= zeta.d2d()
        self.aux_data.fill_BC("coeff")

        coeff_x = myg.scratch_array()
        b = (3, 1, 0, 0)  # this seems more than we need
        coeff_x.v(buf=b)[:,:] = 0.5 * (coeff.ip(-1, buf=b) + coeff.v(buf=b))

        coeff_y = myg.scratch_array()
        b = (0, 0, 3, 1)
        coeff_y.v(buf=b)[:,:] = 0.5 * (coeff.jp(-1, buf=b) + coeff.v(buf=b))

        # we need the MAC velocities on all edges of the computational domain
        # here we do U = U - (zeta/Dh u0) grad (phi/zeta)
        b = (0, 1, 0, 0)
        u_MAC.v(buf=b)[:,:] -= \
                coeff_x.v(buf=b) * (phi_MAC.v(buf=b) - phi_MAC.ip(-1, buf=b)) / myg.dx

        b = (0, 0, 0, 1)
        v_MAC.v(buf=b)[:,:] -= coeff_y.v(buf=b) * \
            (phi_MAC.v(buf=b) - phi_MAC.jp(-1, buf=b)) / myg.dy

        u0_MAC = self.metric.calcu0(u=u_MAC, v=v_MAC)
        #---------------------------------------------------------------------
        # 4. predict D to the edges and do its conservative update
        #---------------------------------------------------------------------

        # FIXME: this is not exactly 4B - be careful with perturbed density
        ldelta_rx = limitFunc(1, D_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_ry = limitFunc(2, D_1.d, myg.qx, myg.qy, myg.ng)
        _rx, _ry = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_rx, ldelta_ry)

        psi_1 = myg.scratch_array(data=scalar_1.d/D_1.d)
        #psi_1.d[:,:] = scalar_1.d / D_1.d
        ldelta_px = limitFunc(1, psi_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_py = limitFunc(2, psi_1.d, myg.qx, myg.qy, myg.ng)
        _px, _py = lm_interface_f.psi_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             psi_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_px, ldelta_py)
        # x component of U0 is zero
        U0_x = myg.scratch_array()
        # is U0 edge-centred?
        _r0x, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                            myg.dx, myg.dy, self.dt,
                                            D0.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                            ldelta_r0x, ldelta_r0y)

        D0_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        D02d = myg.scratch_array() #data=D0.d2df(myg.qx))
        D02d.d[:,:] = D0.d2d()[:,:]
        D02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            (D0_yint.jp(1) * U0_half_star.jp(1)[np.newaxis,:] - \
             D0_yint.v() * U0_half_star.v2d())/myg.dy)

        # predict to edges
        D0_2a_star = Basestate(myg.ny, ng=myg.ng)
        D0_2a_star.d[:] = D0.d
        D0_2a_star.v()[:] = self.lateral_average(D02d.v())

        ldelta_r0x = limitFunc(1, D0_2a_star.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_r0y = limitFunc(2, D0_2a_star.d2df(myg.qx), myg.qx, myg.qy, myg.ng)

        _r0x, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                            myg.dx, myg.dy, self.dt,
                                            D0_2a_star.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                            ldelta_r0x, ldelta_r0y)
        D0_2a_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        D0_2a_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        D_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        D_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        psi_xint = patch.ArrayIndexer(d=_px, grid=myg)
        psi_yint = patch.ArrayIndexer(d=_py, grid=myg)

        D_xint.d[:,:] += 0.5 * (D0_xint.d + D0_2a_xint.d)
        D_yint.d[:,:] += 0.5 * (D0_yint.d + D0_2a_yint.d)

        # FIXME: hacky ceil-/flooring of scalars
        #psi_xint.d[psi_xint.d < 0.] = 0.
        #psi_yint.d[psi_yint.d < 0.] = 0.
        #self.smooth_neighbours(psi_xint, psi_xint.v() > 1.)
        #self.smooth_neighbours(psi_yint, psi_yint.v() > 1.)

        scalar_xint = myg.scratch_array(data=psi_xint.d*D_xint.d)
        #scalar_xint.d[:,:] = psi_xint.d * D_xint.d

        scalar_yint = myg.scratch_array(data=psi_yint.d*D_yint.d)
        #scalar_yint.d[:,:] = psi_yint.d * D_yint.d

        D_old = myg.scratch_array(data=D.d)
        #D_old.d[:,:] = D.d
        scalar_2_star = myg.scratch_array(data=scalar_1.d)
        #scalar_2_star.d[:,:] = scalar_1.d
        D_2_star = myg.scratch_array(data=D_1.d)
        #D_2_star.d[:,:] = D_1.d

        scalar_2_star.v()[:,:] -= self.dt * (
            #  (psi D u)_x
            (scalar_xint.ip(1) * u_MAC.ip(1) - scalar_xint.v() * u_MAC.v())/myg.dx +
            #  (psi D v)_y
            (scalar_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -
             scalar_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy )

        D_2_star.v()[:,:] -= self.dt * (
            #  (D u)_x
            (D_xint.ip(1) * u_MAC.ip(1) - D_xint.v() * u_MAC.v())/myg.dx +
            #  (D v)_y
            (D_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -
             D_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy )

        D0_star = Basestate(myg.ny, ng=myg.ng)
        D0_star.d[:] = D0_2a_star.d
        D0_star.v()[:] = self.lateral_average(D_2_star.v())

        # 4F: compute psi^n+1/2,*
        psi = self.calc_psi(S=S_t_centred, U0=U0_half_star)

        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        # see 4H - need to include a pressure source term here?
        #---------------------------------------------------------------------
        Dh0.v()[:] = self.lateral_average(Dh_1.v())
        ldelta_ex = limitFunc(1, Dh_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_ey = limitFunc(2, Dh_1.d, myg.qx, myg.qy, myg.ng)

        _ex, _ey = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_ex, ldelta_ey)
        _e0x, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh0.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                             ldelta_e0x, ldelta_e0y)

        Dh0_yint = patch.ArrayIndexer(d=_e0y, grid=myg)
        Dh0_xint = patch.ArrayIndexer(d=_e0x, grid=myg)

        Dh02d = myg.scratch_array() #data=Dh0.d2df(myg.qx))
        Dh02d.d[:,:] = Dh0.d2d()
        Dh02d.v()[:,:] += -self.dt * (
            #  (D v)_y
            (Dh0_yint.jp(1) * U0_half_star.jp(1)[np.newaxis,:] - \
             Dh0_yint.v() * U0_half_star.v2d())/myg.dy) + \
            self.dt * u0_MAC.v() * psi.v()

        # predict to edges
        Dh0_star = Basestate(myg.ny, ng=myg.ng)
        Dh0_star.d[:] = Dh0.d
        Dh0_star.v()[:] = self.lateral_average(Dh02d.v())

        ldelta_e0x = limitFunc(1, Dh0_star.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh0_star.d2df(myg.qx), myg.qx, myg.qy, myg.ng)

        _e0x, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                            myg.dx, myg.dy, self.dt,
                                            Dh0_star.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                            ldelta_e0x, ldelta_e0y)
        Dh0_star_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        Dh0_star_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        Dh_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        Dh_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        Dh_xint.d[:,:] += 0.5 * (Dh0_xint.d + Dh0_star_xint.d)
        Dh_yint.d[:,:] += 0.5 * (Dh0_yint.d + Dh0_star_yint.d)

        Dh_old = myg.scratch_array(data=Dh_1.d)
        #Dh_old.d[:,:] = Dh_1.d
        Dh_2_star = myg.scratch_array(data=Dh_1.d)
        #Dh_2_star.d[:,:] = Dh_1.d
        # Dh0 is not edge based?
        drp0 = self.drp0(Dh0=Dh0, u=u_MAC, v=v_MAC, u0=u0_MAC)

        # 4Hii.
        Dh_2_star.v()[:,:] += -self.dt * (
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (Dh_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -\
             Dh_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy ) + \
            self.dt * u0_MAC.v() * v_MAC.v() * drp0.v2d() + \
            self.dt * u0_MAC.v() * psi.v()

        self.cc_data.fill_BC("enthalpy")

        # this makes p0 -> p0_star. May not want to update self.base[p0] here.
        p0 = self.base["p0"]
        p0_star = Basestate(myg.ny, ng=myg.ng)
        p0_star.d[:] = p0.d
        self.enforce_tov(p0=p0_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC, u0=u0_MAC)

        # update eint as a diagnostic
        eint = self.cc_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()


        #---------------------------------------------------------------------
        # 5. React state through dt/2
        #---------------------------------------------------------------------
        D_star = myg.scratch_array(data=D_2_star.d)
        #D_star.d[:,:] = D_2_star.d
        Dh_star = myg.scratch_array(data=Dh_2_star.d)
        #Dh_star.d[:,:] = Dh_2_star.d
        scalar_star = myg.scratch_array(data=scalar_2_star.d)
        #scalar_star.d[:,:] = scalar_2_star.d
        self.react_state(S=self.compute_S(u=u_MAC, v=v_MAC),
                         D=D_star, Dh=Dh_star, scalar=scalar_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC, u0=u0_MAC)

        # FIXME: hacky ceil-/flooring of scalars
        #scalar_star.d[scalar_star.d < 0.] = 0.
        #self.smooth_neighbours(scalar_star, scalar_star.v() > D_star.v())

        #---------------------------------------------------------------------
        # 6. Compute time-centred expasion S, base state velocity U0 and
        # base state forcing
        #---------------------------------------------------------------------
        S_star = self.compute_S(u=u_MAC, v=v_MAC)

        S_half_star = 0.5 * (S + S_star)

        p0_half_star = 0.5 * (p0 + p0_star)

        U0_half = Basestate(myg.ny, ng=myg.ng)
        U0_half.d[:] = U0_half_star.d
        self.compute_base_velocity(U0=U0_half, p0=p0_half_star, S=S_half_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC, u0=u0_MAC)

        #---------------------------------------------------------------------
        # 7. recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        if self.verbose > 0:
            print("  making u, v edge states")
        # FIXME: which Dh are we using here??
        mom_source_x, mom_source_r = self.calc_mom_source(u=u_MAC, v=v_MAC, u0=u0_MAC)
        coeff = self.aux_data.get_var("coeff")
        coeff.d[:,:] = 2.0 / ((Dh.d + Dh_old.d) * u0.d)

        zeta_star = Basestate(myg.ny, ng=myg.ng)
        zeta_star.d[:] = zeta.d
        self.update_zeta(D0=D0_star, zeta=zeta_star, u=u_MAC, v=v_MAC, u0=u0_MAC)
        zeta_half_star = 0.5 * (zeta + zeta_star)
        coeff.d[:,:] *= zeta_half_star.d2d()

        self.aux_data.fill_BC("coeff")

        _ux, _vx, _uy, _vy = \
               lm_interface_f.states(myg.qx, myg.qy, myg.ng,
                                     myg.dx, myg.dy, self.dt,
                                     u.d, v.d,
                                     ldelta_ux, ldelta_vx,
                                     ldelta_uy, ldelta_vy,
                                     coeff.d*gradp_x.d, coeff.d*gradp_y.d,
                                     mom_source_r.d,
                                     u_MAC.d, v_MAC.d)

        u_xint = patch.ArrayIndexer(d=_ux, grid=myg)
        v_xint = patch.ArrayIndexer(d=_vx, grid=myg)
        u_yint = patch.ArrayIndexer(d=_uy, grid=myg)
        v_yint = patch.ArrayIndexer(d=_vy, grid=myg)


        #---------------------------------------------------------------------
        # update U to get the provisional velocity field
        #---------------------------------------------------------------------
        if self.verbose > 0:
            print("  doing provisional update of u, v")

        # compute (U.grad)U

        # we want u_MAC U_x + v_MAC U_y
        advect_x = myg.scratch_array()
        advect_y = myg.scratch_array()

        advect_x.v()[:,:] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(u_xint.ip(1) - u_xint.v())/myg.dx +\
            0.5*(v_MAC.v() + v_MAC.jp(1))*(u_yint.jp(1) - u_yint.v())/myg.dy

        advect_y.v()[:,:] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(v_xint.ip(1) - v_xint.v())/myg.dx +\
            0.5*(v_MAC.v() + v_MAC.jp(1))*(v_yint.jp(1) - v_yint.v())/myg.dy

        proj_type = self.rp.get_param("lm-gr.proj_type")

        if proj_type == 1:
            u.v()[:,:] -= (self.dt * advect_x.v() + self.dt * gradp_x.v())
            v.v()[:,:] -= (self.dt * advect_y.v() + self.dt * gradp_y.v())

        elif proj_type == 2:
            u.v()[:,:] -= self.dt * advect_x.v()
            v.v()[:,:] -= self.dt * advect_y.v()

        # add on source term
        # do we want to use Dh half star here maybe?
        # FIXME: u_MAC, v_MAC in source??
        mom_source_x, mom_source_r = self.calc_mom_source(Dh=Dh_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC, u0=u0_MAC)
        u.d[:,:] += self.dt * mom_source_x.d
        v.d[:,:] += self.dt * mom_source_r.d

        u0 = self.metric.calcu0(u=u, v=v)

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        if self.verbose > 0:
            print("min/max D   = {}, {}".format(self.cc_data.min("density"), self.cc_data.max("density")))
            print("min/max Dh  = {}, {}".format(self.cc_data.min("enthalpy"), self.cc_data.max("enthalpy")))
            print("min/max u   = {}, {}".format(self.cc_data.min("x-velocity"), self.cc_data.max("x-velocity")))
            print("min/max v   = {}, {}".format(self.cc_data.min("y-velocity"), self.cc_data.max("y-velocity")))
            print("min/max psi*D = {}, {}".format(self.cc_data.min("scalar"), self.cc_data.max("scalar")))

        #---------------------------------------------------------------------
        # 8. predict D to the edges and do update
        #---------------------------------------------------------------------

        ldelta_rx = limitFunc(1, D_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0x = limitFunc(1, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_ex = limitFunc(1, Dh_1.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0x = limitFunc(1, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)

        ldelta_ry = limitFunc(2, D_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0y = limitFunc(2, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_ey = limitFunc(2, Dh_1.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)

        _rx, _ry = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_rx, ldelta_ry)
        _r0x, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D0.d2df(myg.qx), U0_x.d,
                                             U0_half.d2df(myg.qx),
                                             ldelta_r0x, ldelta_r0y)

        psi_1.d[:,:] = scalar_1.d / D_1.d
        ldelta_px = limitFunc(1, psi_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_py = limitFunc(2, psi_1.d, myg.qx, myg.qy, myg.ng)
        _px, _py = lm_interface_f.psi_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             psi_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_px, ldelta_py)

        D0_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()
        D02d.v()[:,:] -= self.dt * (
            #  (D v)_y
            (D0_yint.jp(1) * U0_half.jp(1)[np.newaxis,:] -
             D0_yint.v() * U0_half.v2d())/myg.dy)

        # predict to edges
        D0_2a = Basestate(myg.ny, ng=myg.ng)
        D0_2a.d[:] = D0.d
        D0_2a.v()[:] = self.lateral_average(D02d.v())

        ldelta_r0x = limitFunc(1, D0_2a.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_r0y = limitFunc(2, D0_2a.d2df(myg.qx), myg.qx, myg.qy, myg.ng)

        _r0x, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                            myg.dx, myg.dy, self.dt,
                                            D0_2a.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                            ldelta_r0x, ldelta_r0y)
        D0_2a_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        D0_2a_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        D_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        D_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        psi_xint = patch.ArrayIndexer(d=_px, grid=myg)
        psi_yint = patch.ArrayIndexer(d=_py, grid=myg)

        # FIXME: hacky ceil-/flooring of scalars
        #psi_xint.d[psi_xint.d < 0.] = 0.
        #psi_yint.d[psi_yint.d < 0.] = 0.
        #self.smooth_neighbours(psi_xint, psi_xint.v() > 1.)
        #self.smooth_neighbours(psi_yint ,psi_yint.v() > 1.)

        D_xint.d[:,:] += 0.5 * (D0_xint.d + D0_2a_xint.d)
        D_yint.d[:,:] += 0.5 * (D0_yint.d + D0_2a_yint.d)

        scalar_xint.d[:,:] = D_xint.d * psi_xint.d
        scalar_yint.d[:,:] = D_yint.d * psi_yint.d

        D_old = myg.scratch_array(data=D.d)
        #D_old.d[:,:] = D.d
        scalar_2 = myg.scratch_array(data=scalar_1.d)
        #scalar_2.d[:,:] = scalar_1.d
        D_2 = myg.scratch_array(data=D_1.d)
        #D_2.d[:,:] = D_1.d
        D_2.v()[:,:] -= self.dt * (
            #  (D u)_x
            (D_xint.ip(1)*u_MAC.ip(1) - D_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (D_yint.jp(1)*(v_MAC.jp(1) + U0_half.jp(1)[np.newaxis,:]) -\
             D_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy )

        scalar_2.v()[:,:] -= self.dt * (
            #  (D u)_x
            (scalar_xint.ip(1) * u_MAC.ip(1) -
             scalar_xint.v() * u_MAC.v()) / myg.dx +
            #  (D v)_y
            (scalar_yint.jp(1) * (v_MAC.jp(1) +
             U0_half.jp(1)[np.newaxis,:]) -
             scalar_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy )

        # 8D
        D0.v()[:] = self.lateral_average(D_2.v())
        # FIXME: as enforce tov after this, have to use p0_star rather than p0^n+1 here, which seems dodgey?
        # FIXME: also asks for S_half rather than S_half_star, despite this not being calculated?
        psi = self.calc_psi(S=S_half_star, U0=U0_half, old_p0=p0, p0=p0_star)

        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        #---------------------------------------------------------------------
        # CHANGED: this step is in 4G but not 8G, so going to assume that this is a mistake?
        Dh0.v()[:] = self.lateral_average(Dh_1.v())

        _ex, _ey = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_ex, ldelta_ey)
        _e0x, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh0.d2df(myg.qx), U0_x.d, U0_half.d2df(myg.qx),
                                             ldelta_e0x, ldelta_e0y)

        Dh0_xint = patch.ArrayIndexer(d=_e0x, grid=myg)
        Dh0_yint = patch.ArrayIndexer(d=_e0y, grid=myg)

        # FIXME: using the correct u0s here?
        Dh0_old = Basestate(myg.ny, ng=myg.ng)
        Dh0_old.d[:] = Dh0.d

        Dh02d = myg.scratch_array()
        Dh02d.d[:,:] = Dh0.d2d()
        Dh02d.v()[:,:] += -self.dt * (
            #  (D v)_y
            (Dh0_yint.jp(1) * U0_half.jp(1)[np.newaxis,:] -
             Dh0_yint.v() * U0_half.v2d())/myg.dy) + \
            self.dt * u0.v() * psi.v()
        Dh0.v()[:] = self.lateral_average(Dh02d.v())

        # predict to edges
        ldelta_e0x = limitFunc(1, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)

        _e0x, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                            myg.dx, myg.dy, self.dt,
                                            Dh0.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                            ldelta_e0x, ldelta_e0y)
        Dh0_n1_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        Dh0_n1_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        Dh_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        Dh_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        Dh_xint.d[:,:] += 0.5 * (Dh0_xint.d + Dh0_n1_xint.d)
        Dh_yint.d[:,:] += 0.5 * (Dh0_yint.d + Dh0_n1_yint.d)

        Dh_old = myg.scratch_array(data=Dh.d)
        #Dh_old.d[:,:] = Dh.d
        Dh_2 = myg.scratch_array(data=Dh_1.d)
        #Dh_2.d[:,:] = Dh_1.d
        drp0 = self.drp0(Dh0=Dh0, u=u_MAC, v=v_MAC, u0=u0_MAC)

        Dh_2.v()[:,:] += -self.dt * (
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (Dh_yint.jp(1)*(v_MAC.jp(1) + U0_half.jp(1)[np.newaxis,:]) -\
             Dh_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy) + \
            self.dt * u0_MAC.v() * v_MAC.v() * drp0.v2d() + \
            self.dt * u0_MAC.v() * psi.v()

        self.enforce_tov(u=u_MAC, v=v_MAC, u0=u0_MAC)

        # update eint as a diagnostic
        eint = self.cc_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()

        #---------------------------------------------------------------------
        # 9. React state through dt/2
        #---------------------------------------------------------------------
        D.d[:,:] = D_2.d[:,:]
        Dh.d[:,:] = Dh_2.d[:,:]
        scalar.d[:,:] = scalar_2.d[:,:]
        self.react_state(S=self.compute_S(u=u_MAC, v=v_MAC), D=D, Dh=Dh, scalar=scalar, u=u_MAC, v=v_MAC, u0=u0_MAC)

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")
        self.cc_data.fill_BC("scalar")

        #---------------------------------------------------------------------
        # 10. Define the new time expansion S and Gamma1
        #---------------------------------------------------------------------
        oldS = myg.scratch_array(data=S.d)
        #oldS.d[:,:] = S.d

        S = self.compute_S(u=u_MAC, v=v_MAC)

        # moved this here as want to use Dh0^n+1

        #U0_old_half = self.base["U0_old_half"]

        #base_forcing = self.base_state_forcing(U0_half=U0_half, U0_old_half=U0_old_half, Dh0_old=Dh0_old, Dh0=Dh0, u=u, v=v)

        #U0_old_half.d[:] = U0_half.d

        #---------------------------------------------------------------------
        # 11. project the final velocity
        #---------------------------------------------------------------------
        # now we solve L phi = D (U* /dt)
        if self.verbose > 0:
            print("  final projection")

        # create the coefficient array: zeta**2 / Dh u0
        Dh_half = 0.5 * (Dh_old + Dh)
        coeff = 1.0 / (Dh_half * u0)
        zeta_old = Basestate(myg.ny, ng=myg.ng)
        zeta_old.d[:] = zeta.d
        self.update_zeta(u=u_MAC, v=v_MAC, u0=u0_MAC)
        zeta_half = 0.5 * (zeta_old + zeta)
        coeff.d[:,:] *= zeta_half.d2d()**2

        # create the multigrid object
        mg = vcMG.VarCoeffCCMG2d(myg.nx, myg.ny,
                                 xl_BC_type=self.cc_data.BCs["phi"].xlb,
                                 xr_BC_type=self.cc_data.BCs["phi"].xrb,
                                 yl_BC_type=self.cc_data.BCs["phi"].ylb,
                                 yr_BC_type=self.cc_data.BCs["phi"].yrb,
                                 xmin=myg.xmin, xmax=myg.xmax,
                                 ymin=myg.ymin, ymax=myg.ymax,
                                 coeffs=coeff,
                                 coeffs_bc=self.cc_data.BCs["density"],
                                 verbose=0)

        # first compute div{zeta U}

        # u/v are cell-centered, divU is cell-centered
        # this bit seems to use U^n+1 rather than U_MAC
        div_zeta_U.v()[:,:] = \
            0.5 * zeta_half.v2d() * (u.ip(1) - u.ip(-1))/myg.dx + \
            0.5 * (zeta_half.v2dp(1)*v.jp(1) - zeta_half.v2dp(-1)*v.jp(-1))/myg.dy

        # FIXME: check this is using the correct S - might need to be time-centred
        # U or U_MAC??
        constraint = self.constraint_source(zeta=zeta_half)
        mg.init_RHS(div_zeta_U.v(buf=1)/self.dt - constraint.v(buf=1)/self.dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess.v(buf=1)[:,:] = phi.v(buf=1)
        mg.init_solution(phiGuess.d)

        # solve
        mg.solve(rtol=1.e-12, fortran=self.fortran)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi.d[:,:] = mg.get_solution(grid=myg).d

        # get the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)

        # U = U - (zeta/Dh u0) grad (phi)
        # alpha^2 as need to raise grad.
        coeff = 1.0 / (Dh_half * u0)
        coeff.d[:,:] *=  self.metric.alpha.d2d()**2
        coeff.d[:,:] *= zeta_half.d2d()

        # FIXME: need base state forcing here!
        # However, it doesn't actually work: it just causes the atmosphere to rise up?
        u.v()[:,:] += self.dt * (-coeff.v() * gradphi_x.v())
        v.v()[:,:] += self.dt * (-coeff.v() * gradphi_y.v())# + base_forcing.v())

        # store gradp for the next step
        if proj_type == 1:
            gradp_x.v()[:,:] += gradphi_x.v()
            gradp_y.v()[:,:] += gradphi_y.v()

        elif proj_type == 2:
            gradp_x.v()[:,:] = gradphi_x.v()
            gradp_y.v()[:,:] = gradphi_y.v()

        # enforce boundary conditions
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        self.cc_data.fill_BC("gradp_x")
        self.cc_data.fill_BC("gradp_y")

        # FIXME: bcs for base state data
        for var in self.base.values():
            for gz in range(1,myg.ng):
                var.d[myg.jlo-gz] = var.d[myg.jlo]
                var.d[myg.jhi+gz] = var.d[myg.jhi]

        plot_me.d[:,:] =  D.d - D_old.d

        # FIXME: hacky ceil-/flooring of scalars
        #scalar.d[scalar.d < 0.] = 0.
        #self.smooth_neighbours(scalar, scalar.v() > D.v())

        # increment the time
        if not self.in_preevolve:
            self.cc_data.t += self.dt
            self.n += 1


    def dovis(self, vmins=[None, None, None, None], vmaxes=[None, None, None, None]):
        """
        Do runtime visualization
        """
        plt.clf()

        #plt.rc("font", size=10)

        D = self.cc_data.get_var("density")
        scalar = self.cc_data.get_var("scalar")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        #plot_me = self.aux_data.get_var("plot_me")

        myg = self.cc_data.grid

        psi = myg.scratch_array(data=scalar.d/D.d)
        #psi.d[:,:] = scalar.d / D.d

        magvel = np.sqrt(u**2 + v**2)

        vort = myg.scratch_array()

        dv = 0.5 * (v.ip(1) - v.ip(-1)) / myg.dx
        du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy

        vort.v()[:,:] = dv - du

        # FIXME: DELTETEEEEEEEEEE
        #fig, axes = plt.subplots(nrows=1, ncols=1, num=1)
        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.3)

        fields = [D, magvel, psi, vort]
        field_names = [r"$D$", r"$|U|$", r"$\psi$", r"$\nabla\times U$"]
        colourmaps = [plt.cm.jet, plt.cm.jet, plt.cm.seismic,
                      plt.cm.seismic]

        # FIXME: get rid of me!!!!!!!
        #fields = [D]
        #field_names = [r"$D$"]

        #vmins[2] = -1.e-4
        #vmaxes[2] = 1.

        for n in range(len(fields)):
            # FIXME: GET RID OF ME
            ax = axes.flat[n]
            #ax = axes

            f = fields[n]
            cmap = colourmaps[n]

            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            vmin=vmins[n], vmax=vmaxes[n], cmap=cmap)

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(field_names[n])

            plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125,
                    "n: %4d,   t = %10.5f" % (self.n, self.cc_data.t))

        plt.draw()
