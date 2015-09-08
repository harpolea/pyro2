"""
TODO: updateZeta?? What is zeta actually supposed to be? How is it calculated?

TODO: D ln u0/Dt term in momentum equation?

TODO: find out where the slow parts are and speed them up

TODO: u0 takes a while to calculate. Try and identify places where it is calculated multiple times and doesn't need to be.

FIXME: base state boundary conditions

FIXME: check edge/cell-centred/time-centred quantities used correctly

FIXME: add mom_source_x to momentum equation evolution.
In pyro, only have this in the y direction do will need to change the fortran
to allow sourcing in the x direction.

CHANGED: moved tov update in steps 4 and 8 to after the Dh0 update as it is a
function of Dh0

All the keyword arguments of functions default to None as their default values
will be member variables which cannot be accessed in the function's argument
list.
"""

from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt
import pdb

from lm_gr.problems import *
import lm_gr.LM_gr_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
import multigrid.variable_coeff_MG as vcMG
from util import profile
import metric


class Basestate(object):
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
        return self.d[np.newaxis, :]

    def d2df(self, qx):
        """
        fortran compliable version
        """
        return np.array([self.d, ] * qx)

    def v(self, buf=0):
        """
        array without the ghost cells
        """
        return self.d[self.jlo-buf:self.jhi+1+buf]

    def v2d(self, buf=0):
        return self.d[np.newaxis,self.jlo-buf:self.jhi+1+buf]

    def v2df(self, qx, buf=0):
        """
        fortran compliable version
        """
        return np.array(self.d[self.jlo-buf:self.jhi+1+buf, ] * qx)

    def v2dp(self, shift, buf=0):
        """
        2d shifted without ghost cells
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
        return Basestate(self.ny, ng=self.ng, d=self.d.copy())

    def __add__(self, other):
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

    def __init__(self, solver_name, problem_name, rp, timers=None):

        NullSimulation.__init__(self, solver_name, problem_name, rp, timers=timers)

        self.base = {}
        self.aux_data = None
        self.metric = None
        self.dt_old = 1.


    def initialize(self):
        """
        Initialize the grid and variables for low Mach atmospheric flow
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

        my_data.create()

        self.cc_data = my_data

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = patch.CellCenterData2d(myg)

        aux_data.register_var("coeff", bc_dens)
        aux_data.register_var("source_y", bc_yodd)
        aux_data.register_var("old_source_y", bc_yodd)

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
        # FIXME: check whether this is D or rho
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
        oldS = S.copy()


    # This is basically unused now.
    @staticmethod
    def make_prime(a, a0):
        return a - a0.v2d(buf=a0.ng)


    @staticmethod
    def lateral_average(a):
        """
        Calculates and returns the lateral average of a, assuming that stuff is
        to be averaged in the x direction.

        Parameters
        ----------
        a : float array
            2d array to be laterally averaged

        Returns
        -------
        lateralAvg : float array
            lateral average of a
        """
        return np.mean(a, axis=0)


    def update_zeta(self, D0=None, zeta=None, u=None, v=None):
        """
        Updates zeta in the interior and on the edges.
        Assumes all other variables are up to date.
        """

        myg = self.cc_data.grid
        if D0 is None:
            D0 = self.base["D0"]
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
        """
        S = -Gamma^mu_(mu nu) U^nu   (see eq 6.34, 6.37 in LowMachGR).
        base["source-y"] is not updated here as it's sometimes necessary to
        calculate projections of S (e.g. S^n*) and not S^n
        """
        myg = self.cc_data.grid
        S = myg.scratch_array()
        if u is None:
            u = self.cc_data.get_var("x-velocity")
        if v is None:
            v = self.cc_data.get_var("y-velocity")

        chrls = np.array([[self.metric.christoffels([self.cc_data.t, i, j])
                           for j in range(myg.qy)] for i in range(myg.qx)])

        S.d[:,:] = -(chrls[:,:,0,0,0] + chrls[:,:,1,1,0] + chrls[:,:,2,2,0] +
            (chrls[:,:,0,0,1] + chrls[:,:,1,1,1] + chrls[:,:,2,2,1]) * u.d +
            (chrls[:,:,0,0,2] + chrls[:,:,1,1,2] + chrls[:,:,2,2,2]) * v.d)

        return S


    def constraint_source(self, u=None, v=None, S=None, zeta=None):
        """
        calculate the source terms in the constraint, zeta(S - dpdt/ Gamma1 p)

        Returns
        -------
        constraint : float array
            zeta(S - dpdt/ Gamma1 p)
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


    def calc_mom_source(self, u=None, v=None, Dh=None, Dh0=None):
        """
        calculate the source terms in the momentum equation.
        This works only for the metric ds^2 = -a^2 dt^2 + 1/a^2 (dx^2 + dr^2)

        FIXME: need the D_t ln u0 term?

        CHANGED: lowered first index of christoffels

        TODO: make this more general. This definitely needs to be done with
        einsum or something rather than by hand

        Returns
        -------
        mom_source :
            :math:`Gamma_{\rho \nu j} U^\nu U^\rho -
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
        u0 = self.metric.calcu0(u=u, v=v)
        mom_source_r = myg.scratch_array()
        mom_source_x = myg.scratch_array()
        gtt = -(self.metric.alpha.d)**2
        gxx = 1. / self.metric.alpha.d**2
        grr = gxx
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v)

        chrls = np.array([[self.metric.christoffels([self.cc_data.t, i, j])
                           for j in range(myg.qy)] for i in range(myg.qx)])

        # note metric components needed to lower the christoffel symbols
        # NOTE: tried putting D'/D factor in here to see if could drive some movement upwards but it did not do so as desired.
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

        mom_source_x.d[:,:] *=  self.metric.alpha.d2d()**2 * \
       -(Dh0.d2d() - Dh.d) / Dh.d
        mom_source_r.d[:,:] *=  self.metric.alpha.d2d()**2 * \
       -(Dh0.d2d() - Dh.d) / Dh.d
        #pdb.set_trace()

        return mom_source_x, mom_source_r


    def calc_psi(self, S=None, U0=None, p0=None, old_p0=None):
        """
        calculate psi
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


    def base_state_forcing(self, U0_half=None, U0_old_half=None, Dh0_old=None, Dh0=None, u=None, v=None):
        """
        calculate the base state velocity forcing term from 2C
        This works only for the metric ds^2 = -a^2 dt^2 + 1/a^2 (dx^2 + dr^2)
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

        Dh0_half = 0.5 * (Dh0_old + Dh0)

        drp0 = self.drp0(Dh0=Dh0_half, u=u, v=v)
        u0 = self.metric.calcu0(u=u, v=v)
        grr = 1. / self.metric.alpha.d**2

        chrls = np.array([[self.metric.christoffels([self.cc_data.t, i, j])
                           for j in range(myg.qy)] for i in range(myg.qx)])

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


    def react_state(self, S=None, D=None, Dh=None, Dh0=None, u=None, v=None):
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
        if v is None:
            v = self.cc_data.get_var("y-velocity")
        u0 = self.metric.calcu0(u=u, v=v)
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v)
        if S is None:
            S = self.aux_data.get_var("source_y")

        Dh.d[:,:] += 0.5 * self.dt * (S.d * Dh.d + u0.d * v.d * drp0.d)

        D.d[:,:] += 0.5 * self.dt * (S.d * D.d)


    def advect_base_density(self, D0=None, U0=None):
        """
        Updates the base state density through one timestep. Eq. 6.131.
        This is incorrect as need to use edge based D, as found in the
        evolve funciton.
        """
        myg = self.cc_data.grid
        if D0 is None:
            D0 = self.base["D0"]
        dt = self.dt
        dr = myg.dy
        if U0 is None:
            U0 = self.base["U0"]
        # CHANGED: use proper U_0
        # FIXME: time-centred edge states

        D0.v()[:] += -(D0.jp(1) * U0.jp(1) - D0.v() * U0.v()) * dt / dr


    def enforce_tov(self, p0=None, Dh0=None, u=None, v=None):
        """
        enforces the TOV equation. This is the GR equivalent of enforce_hse.
        Eq. 6.132.
        """
        myg = self.cc_data.grid
        if p0 is None:
            p0 = self.base["p0"]
        old_p0 = self.base["old_p0"]
        old_p0 = p0.copy()
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v)

        p0.d[1:] = p0.d[:-1] + 0.5 * self.cc_data.grid.dy * \
                   (drp0.d[1:] + drp0.d[:-1])


    def drp0(self, Dh0=None, u=None, v=None):
        """
        Calculate drp0 as it's messy using eq 6.136
        """
        myg = self.cc_data.grid
        if Dh0 is None:
            Dh0 = self.base["Dh0"]
        # TODO: maybe instead of averaging u0, should calculate it
        # based on U0?
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


    def advect_base_enthalpy(self, Dh0=None, S=None, U0=None, u=None, v=None):
        """
        updates base state enthalpy throung one timestep.
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
        u0 = self.metric.calcu0(u=u, v=v)

        # FIXME: find out how to find the time-centred edge
        # states and use them here?
        # CHANGED: add psi

        Dh0.v()[:] += -(Dh0.jp(1) * U0.jp(1) - Dh0.v() * U0.v()) * dt / dr + \
                      dt * self.lateral_average(u0.v()) * psi.v()
                      # dt * U0.v() * self.drp0().v() + \


    def compute_base_velocity(self, U0=None, p0=None, S=None, Dh0=None, u=None, v=None):
        """
        Caclulates the base velocity using eq. 6.138
        """
        myg = self.cc_data.grid
        if p0 is None:
            p0 = self.base["p0"]
        dt = self.dt
        dr = myg.dy
        if U0 is None:
            U0 = self.base["U0"]
        gamma = self.rp.get_param("eos.gamma")
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v)
        if S is None:
            S = self.aux_data.get_var("source_y")

        # Sbar = latavg(S)
        Sbar = self.lateral_average(S.d)
        U0.d[0] = 0.
        # FIXME: fix cell-centred / edge-centred indexing.
        U0.d[1:] = U0.d[:-1] + dr * (Sbar[:-1] - U0.d[:-1] * drp0.d[:-1] /
                                     (gamma * p0.d[:-1]))


    def compute_timestep(self):
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
        u0 = self.metric.calcu0(u=u, v=v)

        drp0 = self.drp0(Dh0=Dh0, u=u, v=v)
        u01d = Basestate(myg.ny, ng=myg.ng)
        u01d.d[:] = self.lateral_average(u0.d)

        F_buoy = np.max(np.abs(drp0.v() / (Dh0.v() * u01d.v()) ))

        dt_buoy = np.sqrt(2.0 * myg.dx / F_buoy)

        self.dt = min(dt, dt_buoy)
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

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        oldS = self.aux_data.get_var("old_source_y")
        oldS = self.aux_data.get_var("source_y").copy()

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
        mg.solve(rtol=1.e-10)

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
        self.compute_timestep()

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

        # note: the base state quantities do not have valid ghost cells
        self.update_zeta()
        zeta = self.base["zeta"]
        zeta_edges = self.base["zeta-edges"]

        D0 = self.base["D0"]
        Dh0 = self.base["Dh0"]
        U0 = self.base["U0"]

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid

        oldS = self.aux_data.get_var("old_source_y")

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

        ldelta_rx = limitFunc(1, D.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0x = limitFunc(1, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_ex = limitFunc(1, Dh.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0x = limitFunc(1, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_ux = limitFunc(1, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v.d, myg.qx, myg.qy, myg.ng)

        ldelta_ry = limitFunc(2, D.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0y = limitFunc(2, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_ey = limitFunc(2, Dh.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_uy = limitFunc(2, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v.d, myg.qx, myg.qy, myg.ng)

        #---------------------------------------------------------------------
        # 1. React state through dt/2
        #---------------------------------------------------------------------
        D_1 = D.copy()
        Dh_1 = Dh.copy()
        self.react_state(D=D_1, Dh=Dh_1)

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

        U0_half_star = U0.copy()
        self.compute_base_velocity(U0=U0_half_star, S=S_t_centred)

        #pdb.set_trace()

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
        u0 = self.metric.calcu0()
        coeff = self.aux_data.get_var("coeff")
        coeff.d[:,:] = 1.0 / (Dh.d * u0.d)
        # zeta here function of D0^n
        coeff.d[:,:] *= zeta.d2d()
        self.aux_data.fill_BC("coeff")

        g = self.rp.get_param("lm-gr.grav")

        mom_source_x, mom_source_r = self.calc_mom_source(Dh=Dh)

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
        u0 = self.metric.calcu0(u=u, v=v)
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

        constraint = self.constraint_source(u=u_MAC, v=v_MAC, S=S_t_centred)

        # solve the Poisson problem
        mg.init_RHS(div_zeta_U.d - constraint.v(buf=1))
        mg.solve(rtol=1.e-12)

        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/zeta
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC.d[:,:] = mg.get_solution(grid=myg).d
        # this is zero and shouldn't be

        coeff = self.aux_data.get_var("coeff")
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

        #---------------------------------------------------------------------
        # 4. predict D to the edges and do its conservative update
        #---------------------------------------------------------------------
        _rx, _ry = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_rx, ldelta_ry)
        # x component of U0 is zero
        U0_x = myg.scratch_array()
        _, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                            myg.dx, myg.dy, self.dt,
                                            D0.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                            ldelta_r0x, ldelta_r0y)
                                            #D0.d2df(myg.qx), u_MAC.d, v_MAC.d,


        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()[:,:]
        D02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            #(D0_yint.jp(1)*v_MAC.jp(1) - D0_yint.v()*v_MAC.v())/myg.dy)
            (D0_yint.jp(1) * U0_half_star.jp(1)[np.newaxis,:] - \
             D0_yint.v() * U0_half_star.v2d())/myg.dy)
        D0_2a_star = Basestate(myg.ny, ng=myg.ng)
        D0_2a_star.d[:] = self.lateral_average(D02d.d)

        D_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        D_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        D_old = D.copy()
        D_2_star = D_1.copy()

        # CHANGED: isn't there supposed to be a U0 term here?
        D_2_star.v()[:,:] -= self.dt * (
            #  (D u)_x
            (D_xint.ip(1) * u_MAC.ip(1) - D_xint.v() * u_MAC.v())/myg.dx +
            #  (D v)_y
            (D_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -\
             D_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy )


        #print(D_2_star.d[60:70, 60:70])

        self.cc_data.fill_BC("density")

        # 4D Correct D0
        D0_star = Basestate(myg.ny, ng=myg.ng)
        D0_star.d[:] = self.lateral_average(D_2_star.d)

        # 4F: compute psi^n+1/2,*
        psi = self.calc_psi(S=S_t_centred, U0=U0_half_star)

        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        # see 4H - need to include a pressure source term here?
        #---------------------------------------------------------------------
        _ex, _ey = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_ex, ldelta_ey)
        _, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh0.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                             ldelta_e0x, ldelta_e0y)
                                             #Dh0.d2df(myg.qx), u_MAC.d, v_MAC.d,
                                             #ldelta_e0x, ldelta_e0y)

        Dh0_yint = patch.ArrayIndexer(d=_e0y, grid=myg)

        # FIXME: is this is the correct u0?
        Dh0.d[:] = self.lateral_average(Dh_1.d)
        Dh02d = myg.scratch_array()
        Dh02d.d[:,:] = Dh0.d2d()
        Dh02d.v()[:,:] += -self.dt * (
            #  (D v)_y
            (Dh0_yint.jp(1) * U0_half_star.jp(1)[np.newaxis,:] - \
             Dh0_yint.v() * U0_half_star.v2d())/myg.dy) + \
            self.dt * u0.v() * psi.v()
            #(Dh0_yint.jp(1)*v_MAC.jp(1) - Dh0_yint.v()*v_MAC.v())/myg.dy) + \
            #self.dt * u0.v() * psi.v()
        Dh0_star = Dh0.copy()
        Dh0_star.d[:] = self.lateral_average(Dh02d.d)

        Dh_xint = patch.ArrayIndexer(d=_ex, grid=myg)
        Dh_yint = patch.ArrayIndexer(d=_ey, grid=myg)

        Dh_old = Dh_1.copy()
        Dh_2_star = Dh_1.copy()
        # should you use Dh0 or Dh0_star here??
        drp0 = self.drp0(u=u_MAC, v=v_MAC)

        # 4Hii.
        Dh_2_star.v()[:,:] += -self.dt * (
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (Dh_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -\
             Dh_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy ) + \
            self.dt * u0.v() * v_MAC.v() * drp0.v2d() + \
            self.dt * u0.v() * psi.v()

        self.cc_data.fill_BC("enthalpy")

        # this makes p0 -> p0_star. May not want to update self.base[p0] here.
        p0 = self.base["p0"]
        p0_star = p0.copy()
        self.enforce_tov(p0=p0_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC)

        # update eint as a diagnostic
        eint = self.cc_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()


        #---------------------------------------------------------------------
        # 5. React state through dt/2
        #---------------------------------------------------------------------
        D_star = D_2_star.copy()
        Dh_star = Dh_2_star.copy()
        self.react_state(S=self.compute_S(u=u_MAC, v=v_MAC),
                         D=D_star, Dh=Dh_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC)

        #---------------------------------------------------------------------
        # 6. Compute time-centred expasion S, base state velocity U0 and
        # base state forcing
        #---------------------------------------------------------------------
        S_star = self.compute_S(u=u_MAC, v=v_MAC)

        S_half_star = 0.5 * (S + S_star)

        p0_half_star = 0.5 * (p0 + p0_star)

        U0_half = U0_half_star.copy()
        self.compute_base_velocity(U0=U0_half, p0=p0_half_star, S=S_half_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC)

        #pdb.set_trace()

        #---------------------------------------------------------------------
        # 7. recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        if self.verbose > 0:
            print("  making u, v edge states")
        # FIXME: what Dh are we using here??
        mom_source_x, mom_source_r = self.calc_mom_source(u=u_MAC, v=v_MAC)
        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:,:] = 2.0 / ((Dh.v() + Dh_old.v()) * u0.v())

        zeta_star = zeta.copy()
        self.update_zeta(D0=D0_star, zeta=zeta_star, u=u_MAC, v=v_MAC)
        zeta_half_star = 0.5 * (zeta + zeta_star)
        coeff.v()[:,:] *= zeta_half_star.v2d()

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
        mom_source_x, mom_source_r = self.calc_mom_source(Dh=Dh_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC)
        u.d[:,:] += self.dt * mom_source_x.d
        v.d[:,:] += self.dt * mom_source_r.d

        #pdb.set_trace()

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        if self.verbose > 0:
            print("min/max D = {}, {}".format(self.cc_data.min("density"), self.cc_data.max("density")))
            print("min/max Dh = {}, {}".format(self.cc_data.min("enthalpy"), self.cc_data.max("enthalpy")))
            print("min/max u   = {}, {}".format(self.cc_data.min("x-velocity"), self.cc_data.max("x-velocity")))
            print("min/max v   = {}, {}".format(self.cc_data.min("y-velocity"), self.cc_data.max("y-velocity")))

        #---------------------------------------------------------------------
        # 8. predict D to the edges and do update
        #---------------------------------------------------------------------
        _rx, _ry = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_rx, ldelta_ry)
        _, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D0.d2df(myg.qx), U0_x.d,
                                             U0_half.d2df(myg.qx),
                                             ldelta_r0x, ldelta_r0y)

        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        #D0_old = D0.copy()

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()
        D02d.v()[:,:] -= self.dt * (
            #  (D v)_y
            (D0_yint.jp(1) * U0_half.jp(1)[np.newaxis,:] -
             D0_yint.v() * U0_half.v2d())/myg.dy)
        D0_2a = Basestate(myg.ny, ng=myg.ng)
        D0_2a.d[:] = self.lateral_average(D02d.d)

        D_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        D_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        D_old = D.copy()
        D_2 = D_1.copy()

        D_2.v()[:,:] -= self.dt * (
            #  (D u)_x
            (D_xint.ip(1)*u_MAC.ip(1) - D_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (D_yint.jp(1)*(v_MAC.jp(1) + U0_half.jp(1)[np.newaxis,:]) -\
             D_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy )

        # 8D
        D0.d[:] = self.lateral_average(D_2.d)
        # FIXME: as enforce tov after this, have to use p0_star rather than p0^n+1 here, which seems dodgey?
        # FIXME: also asks for S_half rather than S_half_star, despite this not being calculated?
        psi = self.calc_psi(S=S_half_star, U0=U0_half, old_p0=p0, p0=p0_star)

        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        #---------------------------------------------------------------------
        _ex, _ey = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_ex, ldelta_ey)
        _, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh0.d2df(myg.qx), U0_x.d, U0_half.d2df(myg.qx),
                                             ldelta_e0x, ldelta_e0y)

        Dh0_yint = patch.ArrayIndexer(d=_e0y, grid=myg)

        # FIXME: using the correct u0s here?

        Dh0_old = Dh0.copy()

        Dh02d = myg.scratch_array()
        Dh02d.d[:,:] = Dh0.d2d()
        Dh02d.v()[:,:] += -self.dt * (
            #  (D v)_y
            (Dh0_yint.jp(1) * U0_half.jp(1)[np.newaxis,:] -
             Dh0_yint.v() * U0_half.v2d())/myg.dy) + \
            self.dt * u0.v() * psi.v()
        Dh0.d[:] = self.lateral_average(Dh02d.d)

        Dh_xint = patch.ArrayIndexer(d=_ex, grid=myg)
        Dh_yint = patch.ArrayIndexer(d=_ey, grid=myg)

        Dh_old = Dh.copy()
        Dh_2 = Dh_1.copy()
        drp0 = self.drp0(Dh0=Dh0, u=u_MAC, v=v_MAC)

        Dh_2.v()[:,:] += -self.dt * (
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (Dh_yint.jp(1)*(v_MAC.jp(1) + U0_half.jp(1)[np.newaxis,:]) -\
             Dh_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy) + \
            self.dt * u0.v() * v_MAC.v() * drp0.v2d() + \
            self.dt * u0.v() * psi.v()

        self.enforce_tov(u=u_MAC, v=v_MAC)

        # update eint as a diagnostic
        eint = self.cc_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()

        #---------------------------------------------------------------------
        # 9. React state through dt/2
        #---------------------------------------------------------------------
        D = D_2.copy()
        Dh = Dh_2.copy()
        self.react_state(S=self.compute_S(u=u_MAC, v=v_MAC), D=D, Dh=Dh, u=u_MAC, v=v_MAC)

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")

        #---------------------------------------------------------------------
        # 10. Define the new time expansion S and Gamma1
        #---------------------------------------------------------------------
        oldS = S.copy()

        S = self.compute_S(u=u_MAC, v=v_MAC)

        # moved this here as want to use Dh0^n+1

        #U0_old_half = self.base["U0_old_half"]

        #base_forcing = self.base_state_forcing(U0_half=U0_half, U0_old_half=U0_old_half, Dh0_old=Dh0_old, Dh0=Dh0, u=u, v=v)

        #U0_old_half.d[:] = U0_half.d

        #pdb.set_trace()

        #---------------------------------------------------------------------
        # 11. project the final velocity
        #---------------------------------------------------------------------
        # now we solve L phi = D (U* /dt)
        if self.verbose > 0:
            print("  final projection")

        # create the coefficient array: zeta**2 / Dh u0
        Dh_half = 0.5 * (Dh_old + Dh)
        coeff = 1.0 / (Dh_half * u0)
        zeta_old = zeta.copy()
        self.update_zeta(u=u_MAC, v=v_MAC)
        zeta_half = 0.5 * (zeta_old + zeta)
        coeff.v()[:,:] *= zeta_half.v2d()**2

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
        mg.solve(rtol=1.e-12)

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
        coeff.v()[:,:] *= zeta_half.v2d()

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

                # reflect lower boundary, outflow upper
                # var.d[myg.jlo-gz] = var.d[myg.jlo + gz - 1]

        # increment the time
        if not self.in_preevolve:
            self.cc_data.t += self.dt
            self.n += 1


    def dovis(self):
        """
        Do runtime visualization
        """
        plt.clf()

        #plt.rc("font", size=10)

        D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        D0 = self.base["D0"]
        Dprime = self.make_prime(D, D0)

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        myg = self.cc_data.grid

        # make D0 2d
        temp = myg.scratch_array()
        temp.d[:,:] = D0.d[np.newaxis, :]
        D0 = temp

        magvel = np.sqrt(u**2 + v**2)

        vort = myg.scratch_array()

        dv = 0.5 * (v.ip(1) - v.ip(-1)) / myg.dx
        du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy

        #vort.v()[:,:] = dv - du

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.3)

        fields = [D, magvel, v, D0]
        field_names = [r"$D$", r"$|U|$", r"$V$", r"$D_0$"]
        vmins = [98., 0., 0., 98.]
        vmaxes = [101., 0.06, 0.06, 101.]

        for n in range(len(fields)):
            ax = axes.flat[n]

            f = fields[n]

            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            vmin=vmins[n], vmax=vmaxes[n])

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(field_names[n])

            plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125,
                    "n: %4d,   t = %10.5f" % (self.n, self.cc_data.t))

        plt.draw()