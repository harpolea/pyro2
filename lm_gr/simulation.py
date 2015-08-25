"""
TODO: updateZeta?? What is zeta actually supposed to be? How is it calculated?
"""


from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt

from lm_gr.problems import *
import lm_gr.LM_atm_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
import multigrid.variable_coeff_MG as vcMG
from util import profile
import metric


class Basestate(object):
    def __init__(self, ny, ng=0):
        self.ny = ny
        self.ng = ng
        self.qy = ny + 2*ng

        self.d = np.zeros((self.qy), dtype=np.float64)

        self.jlo = ng
        self.jhi = ng+ny-1

    def d2d(self):
        return self.d[np.newaxis, :]

    def v(self, buf=0):
        return self.d[self.jlo-buf:self.jhi+1+buf]

    def v2d(self, buf=0):
        return self.d[np.newaxis,self.jlo-buf:self.jhi+1+buf]

    def v2dp(self, shift, buf=0):
        return self.d[np.newaxis,self.jlo+shift-buf:self.jhi+1+shift+buf]

    def jp(self, shift, buf=0):
        return self.d[self.jlo-buf+shift:self.jhi+1+buf+shift]


class Simulation(NullSimulation):

    def __init__(self, solver_name, problem_name, rp, timers=None):

        NullSimulation.__init__(self, solver_name, problem_name, rp, timers=timers)

        self.base = {}
        self.aux_data = None
        self.metric = None


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

        aux_data.create()
        self.aux_data = aux_data

        # we also need storage for the 1-d base state -- we'll store this
        # in the main class directly.
        self.base["D0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["Dh0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["p0"] = Basestate(myg.ny, ng=myg.ng)

        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.base, self.rp)')
        print("initialised?")

        # Construct beta_0
        gamma = self.rp.get_param("eos.gamma")
        self.base["zeta"] = Basestate(myg.ny, ng=myg.ng)
        self.base["zeta"].d[:] = self.base["p0"].d**(1.0/gamma)

        # we'll also need beta_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        self.base["zeta-edges"] = Basestate(myg.ny, ng=myg.ng)
        self.base["zeta-edges"].jp(1)[:] = \
            0.5*(self.base["zeta"].v() + self.base["zeta"].jp(1))
        self.base["zeta-edges"].d[myg.jlo] = self.base["zeta"].d[myg.jlo]
        self.base["zeta-edges"].d[myg.jhi+1] = self.base["zeta"].d[myg.jhi]

        # add metric
        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")

        alpha = Basestate(myg.ny, ng=myg.ng)
        alpha.d[:] = np.sqrt(1. - 2. * g * (1. - myg.y[:]/R) / c**2)
        beta = [0., 0.]
        gamma_matrix = np.sqrt(1. + 2. * g * (1. - myg.y[:]/R) / c**2) * np.eye(2)
        self.metric = metric.Metric(self.cc_data, self.rp, alpha, beta, gamma_matrix)


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

        try:
            zeta.d()[:] = D0.d / self.lateral_average(u0.d)
        except FloatingPointError:
            print('D0: ', np.max(D0.d))
            print('u0: ', np.max(u0.d))

        # calculate edges
        zeta_edges.jp(1)[:] = 0.5 * (zeta.v() + zeta.jp(1))
        zeta_edges.d[myg.jlo] = zeta.d[myg.jlo]
        zeta_edges.d[myg.jhi+1] = zeta.d[myg.jhi]

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
        v = self.cc_data.get_var("y-velocity")

        S = self.aux_data.get_var("source_y")
        # TODO: compute sourcyness?

        p0 = self.base["p0"]
        dp0dt = Basestate(myg.ny, ng=myg.ng)
        # calculate dp0dt

        constraint = myg.scratch_array()
        # constraint.d[:,:] =
        # TODO: calculate constrait terms

        return constraint


    def compute_timestep(self):
        """
        The timestep() function computes the advective timestep
        (CFL) constraint.  The CFL constraint says that information
        cannot propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        myg = self.cc_data.grid

        cfl = self.rp.get_param("driver.cfl")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # the timestep is min(dx/|u|, dy|v|)
        xtmp = ytmp = 1.e33
        if not abs(u).max() == 0:
            xtmp = myg.dx / abs(u.v()).max()
        if not abs(v).max() == 0:
            ytmp = myg.dy / abs(v.v()).max()

        dt = cfl * min(xtmp, ytmp)

        # We need an alternate timestep that accounts for buoyancy, to
        # handle the case where the velocity is initially zero.
        D = self.cc_data.get_var("density")
        D0 = self.base["D0"]
        Dprime = self.make_prime(D, D0)

        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")

        F_buoy = np.max([(g / (R * c**2)).max(), 1.e-20])

        dt_buoy = np.sqrt(2.0 * myg.dx / F_buoy)

        self.dt = min(dt, dt_buoy)
        if self.verbose > 0: print("timestep is {}".format(dt))


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

        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # the coefficent for the elliptic equation is zeta^2/Dh u0
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
            0.5*zeta.v2d()*(u.ip(1) - u.ip(-1))/myg.dx + \
            0.5*(zeta.v2dp(1)*v.jp(1) - zeta.v2dp(-1)*v.jp(-1))/myg.dy

        # solve D (zeta^2/Dh u0) G (phi/zeta) = D( zeta U )
        constraint = self.constraint_source()
        # set the RHS to divU and solve
        mg.init_RHS(div_zeta_U.d - constraint.d)
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

        if self.verbose > 0: print("done with the pre-evolution")

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


        #---------------------------------------------------------------------
        # create the limited slopes of D, u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("lm-gr.limiter")
        if limiter == 0: limitFunc = reconstruction_f.nolimit
        elif limiter == 1: limitFunc = reconstruction_f.limit2
        else: limitFunc = reconstruction_f.limit4


        ldelta_rx = limitFunc(1, D.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0x = limitFunc(1, D0.d2d(), myg.qx, myg.qy, myg.ng)
        ldelta_ex = limitFunc(1, Dh.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0x = limitFunc(1, Dh0.d2d(), myg.qx, myg.qy, myg.ng)
        ldelta_ux = limitFunc(1, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v.d, myg.qx, myg.qy, myg.ng)

        ldelta_ry = limitFunc(2, D.d, myg.qx, myg.qy, myg.ng)
        ldelta_r0y = limitFunc(2, D0.d2d(), myg.qx, myg.qy, myg.ng)
        ldelta_ey = limitFunc(2, Dh.d,myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh0.d2d(), myg.qx, myg.qy, myg.ng)
        ldelta_uy = limitFunc(2, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v.d, myg.qx, myg.qy, myg.ng)

        # TODO: react_state?

        #---------------------------------------------------------------------
        # get the advective velocities
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
        if self.verbose > 0: print("  making MAC velocities")

        # create the coefficient to the grad (pi/zeta) term
        u0 = self.metric.calcu0()
        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:,:] = 1.0 / (Dh.v() * u0.d)
        coeff.v()[:,:] *= zeta.v2d()
        self.aux_data.fill_BC("coeff")

        # create the source term
        source = self.aux_data.get_var("source_y")

        g = self.rp.get_param("lm-gr.grav")
        c = self.rp.get_param("lm-gr.c")
        R = self.rp.get_param("lm-gr.radius")
        Dprime = self.make_prime(D, D0)
        # TODO: source term?
        # source.v()[:,:] =
        self.aux_data.fill_BC("source_y")

        _um, _vm = lm_interface_f.mac_vels(myg.qx, myg.qy, myg.ng,
                                           myg.dx, myg.dy, self.dt,
                                           u.d, v.d,
                                           ldelta_ux, ldelta_vx,
                                           ldelta_uy, ldelta_vy,
                                           coeff.d*gradp_x.d, coeff.d*gradp_y.d,
                                           source.d)


        u_MAC = patch.ArrayIndexer(d=_um, grid=myg)
        v_MAC = patch.ArrayIndexer(d=_vm, grid=myg)


        #---------------------------------------------------------------------
        # do a MAC projection to make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve D (beta_0^2/D) G phi = D (beta_0 U^MAC), where
        # phi is cell centered, and U^MAC is the MAC-type staggered
        # grid of the advective velocities.

        if self.verbose > 0: print("  MAC projection")

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

        # TODO: update constraint here??

        # solve the Poisson problem
        mg.init_RHS(div_zeta_U.d - contraint.v(buf=1))
        mg.solve(rtol=1.e-12)


        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/beta_0
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC.d[:,:] = mg.get_solution(grid=myg).d

        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:,:] = 1.0 / (Dh.v() * u0.v())
        coeff.v()[:,:] *= zeta.v2d()
        self.aux_data.fill_BC("coeff")

        coeff_x = myg.scratch_array()
        b = (3, 1, 0, 0)  # this seems more than we need
        coeff_x.v(buf=b)[:,:] = 0.5 * (coeff.ip(-1, buf=b) + coeff.v(buf=b))

        coeff_y = myg.scratch_array()
        b = (0, 0, 3, 1)
        coeff_y.v(buf=b)[:,:] = 0.5 * (coeff.jp(-1, buf=b) + coeff.v(buf=b))

        # we need the MAC velocities on all edges of the computational domain
        # here we do U = U - (beta_0/D) grad (phi/beta_0)
        b = (0, 1, 0, 0)
        u_MAC.v(buf=b)[:,:] -= \
                coeff_x.v(buf=b) * (phi_MAC.v(buf=b) - phi_MAC.ip(-1, buf=b)) / myg.dx

        b = (0, 0, 0, 1)
        v_MAC.v(buf=b)[:,:] -= \
                coeff_y.v(buf=b) * (phi_MAC.v(buf=b) - phi_MAC.jp(-1, buf=b)) / myg.dy


        #---------------------------------------------------------------------
        # predict D to the edges and do its conservative update
        #---------------------------------------------------------------------
        _rx, _ry = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D.d, u_MAC.d, v_MAC.d,
                                             ldelta_rx, ldelta_ry)

        _, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             D0.d2d(), u_MAC.d, v_MAC.d,
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

        D0_old = D0.copy()

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()
        D02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            (D0_yint.jp(1)*v_MAC.jp(1) - D0_yint.v()*v_MAC.v())/myg.dy)
        D0.d[:] = self.lateral_average(D02d.d)

        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        #---------------------------------------------------------------------
        _ex, _ey = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh.d, u_MAC.d, v_MAC.d,
                                             ldelta_ex, ldelta_ey)

        _, _e0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             Dh0.d2d(), u_MAC.d, v_MAC.d,
                                             ldelta_e0x, ldelta_e0y)

        Dh_xint = patch.ArrayIndexer(d=_ex, grid=myg)
        Dh_yint = patch.ArrayIndexer(d=_ey, grid=myg)

        Dh_old = Dh.copy()

        Dh.v()[:,:] -= self.dt*(
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/myg.dx +
            #  (D v)_y
            (Dh_yint.jp(1)*v_MAC.jp(1) - Dh_yint.v()*v_MAC.v())/myg.dy )

        self.cc_data.fill_BC("density")

        Dh0_yint = patch.ArrayIndexer(d=_e0y, grid=myg)

        Dh0_old = Dh0.copy()

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

        # TODO: update bcs? enforce HSE?

        #---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        if self.verbose > 0:
            print("  making u, v edge states")

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
                                     coeff.d * gradp_x.d, coeff.d * gradp_y.d,
                                     source.d,
                                     u_MAC.d, v_MAC.d)

        u_xint = patch.ArrayIndexer(d=_ux, grid=myg)
        v_xint = patch.ArrayIndexer(d=_vx, grid=myg)
        u_yint = patch.ArrayIndexer(d=_uy, grid=myg)
        v_yint = patch.ArrayIndexer(d=_vy, grid=myg)


        #---------------------------------------------------------------------
        # update U to get the provisional velocity field
        #---------------------------------------------------------------------
        if self.verbose > 0: print("  doing provisional update of u, v")

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
            u.v()[:,:] -= (self.dt*advect_x.v() + self.dt*gradp_x.v())
            v.v()[:,:] -= (self.dt*advect_y.v() + self.dt*gradp_y.v())

        elif proj_type == 2:
            u.v()[:,:] -= self.dt * advect_x.v()
            v.v()[:,:] -= self.dt * advect_y.v()


        # add the gravitational source
        # TODO: check this for gr case
        u0 = self.metric.calcu0()
        D_half = 0.5*(D + D_old)
        Dprime = self.make_prime(D_half, D0)
        source.d[:,:] = (Dprime*g/D_half).d
        self.aux_data.fill_BC("source_y")

        v.d[:,:] += self.dt*source.d

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        if self.verbose > 0:
            print("min/max D = {}, {}".format(self.cc_data.min("density"), self.cc_data.max("density")))
            print("min/max u   = {}, {}".format(self.cc_data.min("x-velocity"), self.cc_data.max("x-velocity")))
            print("min/max v   = {}, {}".format(self.cc_data.min("y-velocity"), self.cc_data.max("y-velocity")))

        # TODO: react state?


        #---------------------------------------------------------------------
        # project the final velocity
        #---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)
        if self.verbose > 0: print("  final projection")

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

        constraint = self.constraint_source()
        mg.init_RHS(div_zeta_U.d/self.dt - constraint.d/self.dt)

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

        dv = 0.5*(v.ip(1) - v.ip(-1))/myg.dx
        du = 0.5*(u.jp(1) - u.jp(-1))/myg.dy

        vort.v()[:,:] = dv - du

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.25)

        fields = [D, magvel, vort, Dprime]
        field_names = [r"$D$", r"|U|", r"$\nabla \times U$", r"$D'$"]

        for n in range(len(fields)):
            ax = axes.flat[n]

            f = fields[n]

            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(field_names[n])

            #plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()
