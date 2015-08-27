"""
TODO: updateZeta?? What is zeta actually supposed to be? How is it calculated?

TODO: D ln u0/Dt term in momentum equation?

TODO: find out where the slow parts are and speed them up
"""


from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt

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
        self.qy = ny + 2*ng

        if d is None:
            self.d = np.zeros((self.qy), dtype=np.float64)
        else:
            self.d = d

        self.jlo = ng
        self.jhi = ng+ny-1

    def d2d(self):
        return self.d[np.newaxis, :]

    def d2df(self, qx):
        """
        fortran compliable version
        """
        return np.array([self.d, ] * qx)

    def v(self, buf=0):
        return self.d[self.jlo-buf:self.jhi+1+buf]

    def v2d(self, buf=0):
        return self.d[np.newaxis,self.jlo-buf:self.jhi+1+buf]

    def v2df(self, qx, buf=0):
        """
        fortran compliable version
        """
        return np.array(self.d[self.jlo-buf:self.jhi+1+buf, ] * qx)

    def v2dp(self, shift, buf=0):
        return self.d[np.newaxis,self.jlo+shift-buf:self.jhi+1+shift+buf]

    def v2dpf(self, qx, shift, buf=0):
        """
        fortran compliable version
        """
        return np.array(self.d[self.jlo+shift-buf:self.jhi+1+shift+buf, ] * qx)

    def jp(self, shift, buf=0):
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
        self.old_dt = 1.


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
        for bc in [self.rp.get_param("mesh.xlboundary"),
                   self.rp.get_param("mesh.xrboundary"),
                   self.rp.get_param("mesh.ylboundary"),
                   self.rp.get_param("mesh.yrboundary")]:
            if bc == "periodic":
                bctype = "periodic"
            elif bc in ["reflect", "slipwall"]:
                bctype = "neumann"
            elif bc in ["outflow"]:
                bctype = "dirichlet"
            bcs.append(bctype)

        bc_phi = patch.BCObject(xlb=bcs[0], xrb=bcs[1], ylb=bcs[2], yrb=bcs[3])

        my_data.register_var("phi-MAC", bc_phi)
        my_data.register_var("phi", bc_phi)


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

        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.base, self.rp)')

        # add metric
        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")

        alpha = Basestate(myg.ny, ng=myg.ng)
        alpha.d[:] = np.sqrt(1. - 2. * g * (1. - myg.y[:]/R) / c**2)
        beta = [0., 0.]
        gamma_matrix = np.sqrt(1. + 2. * g / c**2) * np.eye(2)
        self.metric = metric.Metric(self.cc_data, self.rp, alpha, beta, gamma_matrix)
        u0 = self.metric.calcu0()

        # Construct zeta
        # FIXME: what is zeta supposed to be???
        gamma = self.rp.get_param("eos.gamma")
        self.base["zeta"] = Basestate(myg.ny, ng=myg.ng)
        #self.base["zeta"].d[:] = self.base["p0"].d**(1.0/gamma)
        D0 = self.base["D0"]
        self.base["zeta"].d[:] = D0.d / self.lateral_average(u0.d)

        # we'll also need zeta on vertical edges -- on the domain edges,
        # just do piecewise constant
        self.base["zeta-edges"] = Basestate(myg.ny, ng=myg.ng)
        self.base["zeta-edges"].jp(1)[:] = \
            0.5*(self.base["zeta"].v() + self.base["zeta"].jp(1))
        self.base["zeta-edges"].d[myg.jlo] = self.base["zeta"].d[myg.jlo]
        self.base["zeta-edges"].d[myg.jhi+1] = self.base["zeta"].d[myg.jhi]

        # initialise source
        S = self.aux_data.get_var("source_y")
        S = self.compute_S()
        oldS = self.aux_data.get_var("old_source_y")
        oldS = S.copy()


    @staticmethod
    def make_prime(a, a0):
        return a - a0.v2d(buf=a0.ng)


    @staticmethod
    def lateral_average(a):
        """
        Calculates and returns the lateral average of a, assuming that stuff is to be averaged in the x direction.

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


    def update_zeta(self):
        """
        Updates zeta in the interior and on the edges. Assumes all other variables are up to date.
        """

        myg = self.cc_data.grid
        D0 = self.base["D0"]
        u0 = self.metric.calcu0()

        zeta = self.base["zeta"]
        zeta_edges = self.base["zeta-edges"]

        # FIXME: how do you calculate zeta??
        try:
            zeta.d[:] = D0.d / self.lateral_average(u0.d)
        except FloatingPointError:
            print('D0: ', np.max(D0.d))
            print('u0: ', np.max(u0.d))

        # calculate edges
        zeta_edges.jp(1)[:] = 0.5 * (zeta.v() + zeta.jp(1))
        zeta_edges.d[myg.jlo] = zeta.d[myg.jlo]
        zeta_edges.d[myg.jhi+1] = zeta.d[myg.jhi]


    def compute_S(self):
        """
        S = -Gamma^mu_(mu nu) U^nu   (see eq 6.34, 6.37 in LowMachGR).
        base["source-y"] is not updated here as it's sometimes necessary to
        calculate projections and not S^n
        """
        myg = self.cc_data.grid
        S = myg.scratch_array()
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # TODO: slicing rather than looping
        for i in range(myg.qx):
            for j in range(myg.qy):
                chrls = self.metric.christoffels([self.cc_data.t, i,j])
                S.d[i,j] = -(chrls[0,0,0] + chrls[1,1,0] + chrls[2,2,0] +
                    (chrls[0,0,1] + chrls[1,1,1] + chrls[2,2,1]) * u.d[i,j] +
                    (chrls[0,0,2] + chrls[1,1,2] + chrls[2,2,2]) * v.d[i,j])

        return S


    def constraint_source(self):
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
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        zeta = self.base["zeta"]
        S = self.aux_data.get_var("source_y")

        p0 = self.base["p0"]
        dp0dt = Basestate(myg.ny, ng=myg.ng)
        # calculate dp0dt
        # FIXME: assumed it's 0 for now

        constraint = myg.scratch_array()
        constraint.d[:,:] = zeta.d2df(myg.qx) * (S.d - dp0dt.d2df(myg.qx) / (gamma * p0.d2df(myg.qx)))

        return constraint


    def react_state(self):
        """
        gravitational source terms in the continuity equation (called react state to mirror MAESTRO as here they just have source terms from the reactions)
        """
        myg = self.cc_data.grid

        D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        u0 = self.metric.calcu0()
        drp0 = self.drp0()
        S = self.aux_data.get_var("source_y")

        # source is always zero?
        #print('source: ', S.d[0:10, 0:10])

        Dh.v()[:,:] += 0.5 * self.dt * (S.v() * Dh.v() +
                                        u0.v() * v.v() * drp0.v())

        D.v()[:,:] += 0.5 * self.dt * (S.v() * D.v())


    def advect_base_density(self, D0=None, U0=None):
        """
        Updates the base state density through one timestep. Eq. 6.130.
        """
        myg = self.cc_data.grid
        if D0 is None:
            D0 = self.base["D0"]
        dt = self.dt
        dr = myg.dy
        if U0 is None:
            U0 = self.base["U0"]
        # FIXME: use proper U_0 and time-centred edge states.

        D0.v()[:] += -(D0.jp(1) * U0.jp(1) - D0.v() * U0.v()) * dt / dr


    def enforce_tov(self, p0=None):
        """
        enforces the TOV equation. This is the GR equivalent of enforce_hse. Eq. 6.132.
        """
        if p0 is None:
            p0 = self.base["p0"]
        old_p0 = self.base["old_p0"]
        old_p0 = p0.copy()
        drp0 = self.drp0()

        p0.jp(1, buf=1)[:] = p0.v(buf=1) + 0.5 * self.cc_data.grid.dy * \
                             (drp0.jp(1, buf=1) + drp0.v(buf=1))


    def drp0(self):
        """
        Calculate drp0 as it's messy using eq 6.135
        """
        myg = self.cc_data.grid
        p0 = self.base["p0"]
        Dh0 = self.base["Dh0"]
        u0 = self.metric.calcu0()
        u01d = Basestate(myg.ny, ng=myg.ng)
        u01d.d[:] = self.lateral_average(u0.d)
        alpha = self.metric.alpha
        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")

        drp0 = Basestate(myg.ny, ng=myg.ng)

        drp0.d[:] = g * (Dh0.d / u01d.d + 2. * p0.d * (1. - alpha.d**4)) / \
              (R * c**2 * alpha.d**2)

        return drp0


    def advect_base_enthalpy(self, Dh0=None, U0=None):
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

        # FIXME: calculate U_0, find out how to find the time-centred edge states and use them here.

        Dh0.v()[:] += -(Dh0.jp(1) * U0.jp(1) - Dh.v() * U0.v()) * dt / dr + \
                      dt * U0.v() * self.drp0().v()


    def compute_base_velocity(self, p0=None, S=None):
        """
        Caclulates the base velocity using eq. 6.137
        """
        myg = self.cc_data.grid
        if p0 is None:
            p0 = self.base["p0"]
        dt = self.dt
        dr = myg.dy
        U0 = self.base["U0"]
        gamma = self.rp.get_param("eos.gamma")
        drp0 = self.drp0()
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        if S is None:
            S = self.aux_data.get_var("source_y")

        # Sbar = latavg(S)
        Sbar = self.lateral_average(S.d)
        U0.d[0] = 0.
        # FIXME: fix cell-centred / edge-centred indexing.
        for i in range(myg.qy-1):
            U0.d[i+1] = U0.d[i] + dr * (Sbar[i] - U0.d[i] * drp0.d[i] /
                                        (gamma * p0.d[i]))


    def compute_timestep(self):
        """
        The timestep() function computes the advective timestep
        (CFL) constraint.  The CFL constraint says that information
        cannot propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        self.old_dt = self.dt

        myg = self.cc_data.grid

        cfl = self.rp.get_param("driver.cfl")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # the timestep is min(dx/|u|, dy|v|)
        xtmp = ytmp = 1.e33
        if not abs(u).max() < 1e-25:
            xtmp = myg.dx / abs(u.v()).max()
        if not abs(v).max() < 1e-25:
            ytmp = myg.dy / abs(v.v()).max()

        dt = cfl * min(xtmp, ytmp)

        # We need an alternate timestep that accounts for buoyancy, to
        # handle the case where the velocity is initially zero.
        Dh0 = Dh0 = self.base["Dh0"]
        u0 = self.metric.calcu0()

        # FIXME: do this properly for gr case
        drp0 = self.drp0()
        u01d = Basestate(myg.ny, ng=myg.ng)
        u01d.d[:] = self.lateral_average(u0.d)

        F_buoy = np.max(drp0.v() / (Dh0.v() * u01d.v()))

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
            0.5*zeta.v2df(myg.qx)*(u.ip(1) - u.ip(-1))/myg.dx + \
            0.5*(zeta.v2dpf(myg.qx, 1)*v.jp(1) - \
            zeta.v2dpf(myg.qx, -1)*v.jp(-1))/myg.dy

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

        coeff = 1. / (Dh * u0)
        coeff.v()[:,:] *= zeta.v2d()

        u.v()[:,:] -= coeff.v() * gradp_x.v()
        v.v()[:,:] -= coeff.v() * gradp_y.v()

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 2. now get an approximation to gradp at n-1/2 by going through the
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

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid

        oldS = self.aux_data.get_var("old_source_y")

        #---------------------------------------------------------------------
        # create the limited slopes of D, u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("lm-gr.limiter")
        if limiter == 0: limitFunc = reconstruction_f.nolimit
        elif limiter == 1: limitFunc = reconstruction_f.limit2
        else: limitFunc = reconstruction_f.limit4

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
        self.react_state()

        #---------------------------------------------------------------------
        # 2. Compute provisional S, U0 and base state forcing
        #---------------------------------------------------------------------
        S = self.aux_data.get_var("source_y")
        S_t_centred = myg.scratch_array()
        S = self.aux_data.get_var("source_y")
        if self.cc_data.t == 0:
            S_t_centred.d[:,:] = 0.5 * (oldS.d + S.d)
        else:
            S_t_centred.d[:,:] = S.d + self.dt * 0.5 * (S.d - oldS.d) / self.old_dt

        self.compute_base_velocity(S=S_t_centred)

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
        u0 = self.metric.calcu0()
        coeff = self.aux_data.get_var("coeff")
        coeff.d[:,:] = 1.0 / (Dh.d * u0.d)
        coeff.d[:,:] *= zeta.d2d()
        self.aux_data.fill_BC("coeff")

        # create source term
        # this is problematic in the gr case as this source term is a vector:
        # in Newtonian case, it's a scalar (times the radial normal vector)
        # FIXME: for now, shall just calculate radial component and wish for the # best.
        # FIXME: not sure this is the right source term for the elliptic?
        mom_source = myg.scratch_array()

        # TODO: slicing rather than looping
        for i in range(myg.qx):
            for j in range(myg.qy):
                chrls = self.metric.christoffels([self.cc_data.t, i,j])
                mom_source.d[i,j] = -(chrls[0,0,2] +
                (chrls[1,0,2] + chrls[0,1,2]) * u.d[i,j] +
                (chrls[2,0,2] + chrls[0,2,2]) * v.d[i,j] +
                chrls[1,1,2] * u.d[i,j]**2 + chrls[2,2,2] * v.d[i,j]**2 +
                (chrls[2,1,2] + chrls[1,2,2]) * u.d[i,j] * v.d[i,j])


        g = self.rp.get_param("lm-gr.grav")
        #c = self.rp.get_param("lm-gr.c")
        #R = self.rp.get_param("lm-gr.radius")
        Dprime = self.make_prime(D, D0)
        _um, _vm = lm_interface_f.mac_vels(myg.qx, myg.qy, myg.ng,
                                           myg.dx, myg.dy, self.dt,
                                           u.d, v.d,
                                           ldelta_ux, ldelta_vx,
                                           ldelta_uy, ldelta_vy,
                                           coeff.d*gradp_x.d,
                                           coeff.d*gradp_y.d,
                                           S.d)

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

        # create the coefficient array: zeta**2/D
        # MZ!!!! probably don't need the buf here
        # TODO: are zeta, u0 functions of MAC velocities?
        coeff.v(buf=1)[:,:] = 1. / (Dh.v(buf=1) * u0.v(buf=1))
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

        constraint = self.constraint_source()

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
        coeff.d[:,:] = 1.0 / (Dh.d * u0.d)
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
        v_MAC.v(buf=b)[:,:] -= \
                coeff_y.v(buf=b) * (phi_MAC.v(buf=b) - phi_MAC.jp(-1, buf=b)) / myg.dy


        #---------------------------------------------------------------------
        # 4. predict D to the edges and do its conservative update
        #---------------------------------------------------------------------
        _rx, _ry = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D.d, u_MAC.d, v_MAC.d,
                                             ldelta_rx, ldelta_ry)
        _, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D0.d2df(myg.qx), u_MAC.d, v_MAC.d,
                                             ldelta_r0x, ldelta_r0y)

        D_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        D_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        D_old = D.copy()

        D.v()[:,:] -= self.dt*(
            #  (D u)_x
            (D_xint.ip(1)*u_MAC.ip(1) - D_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (D_yint.jp(1)*v_MAC.jp(1) - D_yint.v()*v_MAC.v())/myg.dy )

        self.cc_data.fill_BC("density")

        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        #D0_old = D0.copy()

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()
        D02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            (D0_yint.jp(1)*v_MAC.jp(1) - D0_yint.v()*v_MAC.v())/myg.dy)
        D0.d[:] = self.lateral_average(D02d.d)

        self.enforce_tov()

        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        # see 4H - need to include a pressure source term here?
        #---------------------------------------------------------------------
        _ex, _ey = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh.d, u_MAC.d, v_MAC.d,
                                             ldelta_ex, ldelta_ey)
        _, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh0.d2df(myg.qx), u_MAC.d, v_MAC.d,
                                             ldelta_e0x, ldelta_e0y)

        Dh_xint = patch.ArrayIndexer(d=_ex, grid=myg)
        Dh_yint = patch.ArrayIndexer(d=_ey, grid=myg)

        Dh_old = Dh.copy()
        drp0 = self.drp0()

        # 4Hii.
        # FIXME: need to add on psi term.
        Dh.v()[:,:] += -self.dt*(
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (Dh_yint.jp(1)*v_MAC.jp(1) - Dh_yint.v()*v_MAC.v())/myg.dy ) + \
            self.dt * u0.v() * v_MAC.v() * drp0.v2d()


        self.cc_data.fill_BC("enthalpy")

        Dh0_yint = patch.ArrayIndexer(d=_e0y, grid=myg)

        #Dh0_old = Dh0.copy()

        # FIXME: need a u0 psi term here
        Dh02d = myg.scratch_array()
        Dh02d.d[:,:] = Dh0.d2d()
        Dh02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            (Dh0_yint.jp(1)*v_MAC.jp(1) - Dh0_yint.v()*v_MAC.v())/myg.dy)
        Dh0.d[:] = self.lateral_average(Dh02d.d)

        # update eint as a diagnostic
        eint = self.cc_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()


        #---------------------------------------------------------------------
        # 5. React state through dt/2
        #---------------------------------------------------------------------
        self.react_state()

        #---------------------------------------------------------------------
        # 6. Compute time-centred expasion S, base state velocity U0 and
        # base state forcing
        #---------------------------------------------------------------------
        S_star = self.compute_S()

        S_half_star = 0.5 * (S + S_star)

        old_p0 = self.base["old_p0"]
        p0 = self.base["p0"]
        p0_half_star = 0.5 * (p0 + old_p0)

        self.compute_base_velocity(p0=p0_half_star, S=S_half_star)

        #---------------------------------------------------------------------
        # 7. recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        if self.verbose > 0:
            print("  making u, v edge states")

        # create source term
        # have used MAC velocities here?
        # TODO: slicing rather than looping
        for i in range(myg.qx):
            for j in range(myg.qy):
                chrls = self.metric.christoffels([self.cc_data.t, i,j])
                mom_source.d[i,j] = -(chrls[0,0,2] +
                (chrls[1,0,2] + chrls[0,1,2]) * u_MAC.d[i,j] +
                (chrls[2,0,2] + chrls[0,2,2]) * v_MAC.d[i,j] +
                chrls[1,1,2] * u_MAC.d[i,j]**2 +
                chrls[2,2,2] * v_MAC.d[i,j]**2 +
                (chrls[2,1,2] + chrls[1,2,2]) * u_MAC.d[i,j] * v_MAC.d[i,j])

        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:,:] = 2.0 / ((Dh.v() + Dh_old.v()) * u0.v())
        coeff.v()[:,:] *= zeta.v2d()
        self.aux_data.fill_BC("coeff")
        _ux, _vx, _uy, _vy = \
               lm_interface_f.states(myg.qx, myg.qy, myg.ng,
                                     myg.dx, myg.dy, self.dt,
                                     u.d, v.d,
                                     ldelta_ux, ldelta_vx,
                                     ldelta_uy, ldelta_vy,
                                     coeff.d*gradp_x.d, coeff.d*gradp_y.d,
                                     S.d,
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
        #print(v_yint.d[20:30, 20:30])

        proj_type = self.rp.get_param("lm-gr.proj_type")

        if proj_type == 1:
            u.v()[:,:] -= (self.dt * advect_x.v() + self.dt * gradp_x.v())
            v.v()[:,:] -= (self.dt * advect_y.v() + self.dt * gradp_y.v())

        elif proj_type == 2:
            u.v()[:,:] -= self.dt * advect_x.v()
            v.v()[:,:] -= self.dt * advect_y.v()


        # add the gravitational source
        # TODO: check this for gr case
        #u0 = self.metric.calcu0()
        #D_half = 0.5 * (D + D_old)
        #Dprime = self.make_prime(D_half, D0)
        #source.d[:,:] = (Dprime * g / D_half).d
        #self.aux_data.fill_BC("source_y")

        # TODO: slicing rather than looping
        for i in range(myg.qx):
            for j in range(myg.qy):
                chrls = self.metric.christoffels([self.cc_data.t, i,j])
                mom_source.d[i,j] = -(chrls[0,0,2] +
                (chrls[1,0,2] + chrls[0,1,2]) * u.d[i,j] +
                (chrls[2,0,2] + chrls[0,2,2]) * v.d[i,j] +
                chrls[1,1,2] * u.d[i,j]**2 + chrls[2,2,2] * v.d[i,j]**2 +
                (chrls[2,1,2] + chrls[1,2,2]) * u.d[i,j] * v.d[i,j])

        v.d[:,:] += self.dt * mom_source.d

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
                                             D.d, u_MAC.d, v_MAC.d,
                                             ldelta_rx, ldelta_ry)
        _, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D0.d2df(myg.qx), u_MAC.d, v_MAC.d,
                                             ldelta_r0x, ldelta_r0y)

        D_xint = patch.ArrayIndexer(d=_rx, grid=myg)
        D_yint = patch.ArrayIndexer(d=_ry, grid=myg)

        D_old = D.copy()

        D.v()[:,:] -= self.dt*(
            #  (D u)_x
            (D_xint.ip(1)*u_MAC.ip(1) - D_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (D_yint.jp(1)*v_MAC.jp(1) - D_yint.v()*v_MAC.v())/myg.dy )

        self.cc_data.fill_BC("density")

        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        #D0_old = D0.copy()

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()
        D02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            (D0_yint.jp(1)*v_MAC.jp(1) - D0_yint.v()*v_MAC.v())/myg.dy)
        D0.d[:] = self.lateral_average(D02d.d)

        self.enforce_tov()

        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        #---------------------------------------------------------------------
        _ex, _ey = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh.d, u_MAC.d, v_MAC.d,
                                             ldelta_ex, ldelta_ey)
        _, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh0.d2df(myg.qx), u_MAC.d, v_MAC.d,
                                             ldelta_e0x, ldelta_e0y)

        Dh_xint = patch.ArrayIndexer(d=_ex, grid=myg)
        Dh_yint = patch.ArrayIndexer(d=_ey, grid=myg)

        Dh_old = Dh.copy()

        Dh.v()[:,:] -= self.dt*(
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (Dh_yint.jp(1)*v_MAC.jp(1) - Dh_yint.v()*v_MAC.v())/myg.dy )

        self.cc_data.fill_BC("enthalpy")

        Dh0_yint = patch.ArrayIndexer(d=_e0y, grid=myg)

        #Dh0_old = Dh0.copy()

        Dh02d = myg.scratch_array()
        Dh02d.d[:,:] = Dh0.d2d()
        Dh02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            (Dh0_yint.jp(1)*v_MAC.jp(1) - Dh0_yint.v()*v_MAC.v())/myg.dy)
        Dh0.d[:] = self.lateral_average(Dh02d.d)

        # update eint as a diagnostic
        eint = self.cc_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()

        #---------------------------------------------------------------------
        # 9. React state through dt/2
        #---------------------------------------------------------------------
        self.react_state()

        #---------------------------------------------------------------------
        # 10. Define the new time expansion S and Gamma1
        #---------------------------------------------------------------------
        oldS = S.copy()

        S = self.compute_S()

        #---------------------------------------------------------------------
        # 11. project the final velocity
        #---------------------------------------------------------------------
        # now we solve L phi = D (U* /dt)
        if self.verbose > 0:
            print("  final projection")

        # create the coefficient array: zeta**2 / Dh u0
        coeff = 1.0 / (Dh * u0)
        self.update_zeta()
        coeff.v()[:,:] *= zeta.v2d()**2

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
        div_zeta_U.v()[:,:] = \
            0.5 * zeta.v2d() * (u.ip(1) - u.ip(-1))/myg.dx + \
            0.5 * (zeta.v2dp(1)*v.jp(1) - zeta.v2dp(-1)*v.jp(-1))/myg.dy
        # FIXME: this is zero always as u, v are zero

        constraint = self.constraint_source()
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
        # FIXME: this is always zero at the moment?
        #print(phi.d[0:10, 0:10])

        # get the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)

        # U = U - (zeta/Dh u0) grad (phi)
        coeff = 1.0 / (Dh * u0)
        coeff.v()[:,:] *= zeta.v2d()

        u.v()[:,:] -= self.dt * coeff.v() * gradphi_x.v()
        v.v()[:,:] -= self.dt * coeff.v() * gradphi_y.v()

        # store gradp for the next step

        if proj_type == 1:
            gradp_x.v()[:,:] += gradphi_x.v()
            gradp_y.v()[:,:] += gradphi_y.v()

        elif proj_type == 2:
            gradp_x.v()[:,:] = gradphi_x.v()
            gradp_y.v()[:,:] = gradphi_y.v()

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        self.cc_data.fill_BC("gradp_x")
        self.cc_data.fill_BC("gradp_y")

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
        D0 = self.base["D0"]
        Dprime = self.make_prime(D, D0)

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        myg = self.cc_data.grid

        magvel = np.sqrt(u**2 + v**2)

        vort = myg.scratch_array()

        dv = 0.5 * (v.ip(1) - v.ip(-1))/myg.dx
        du = 0.5 * (u.jp(1) - u.jp(-1))/myg.dy

        vort.v()[:,:] = dv - du

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.25)

        fields = [D, magvel, vort, Dprime]
        field_names = [r"$D$", r"$|U|$", r"$\nabla \times U$", r"$D'$"]

        for n in range(len(fields)):
            ax = axes.flat[n]

            f = fields[n]

            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(field_names[n])

            plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125, "n: %4d,   t = %10.5f" % (self.n, self.cc_data.t))

        plt.draw()
