from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from lm_atm.problems import *
import lm_atm.LM_atm_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
import multigrid.variable_coeff_MG as vcMG
from util import profile

class Simulation:

    def __init__(self, problem_name, rp, timers=None):
        """
        Initialize the Simulation object for incompressible flow.

        Parameters
        ----------
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in LM-atmosphere/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        """

        self.rp = rp

        print(rp)

        self.cc_data = None
        self.aux_data = None
        self.base = {}
        self.metric = None

        self.problem_name = problem_name

        if timers == None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers


    def initialize(self):
        """
        Initialize the grid and variables for low Mach atmospheric flow
        and set the initial conditions for the chosen problem.
        """

        # setup the grid
        nx = self.rp.get_param("mesh.nx")
        ny = self.rp.get_param("mesh.ny")

        xmin = self.rp.get_param("mesh.xmin")
        xmax = self.rp.get_param("mesh.xmax")
        ymin = self.rp.get_param("mesh.ymin")
        ymax = self.rp.get_param("mesh.ymax")

        myg = patch.Grid2d(nx, ny,
                           xmin=xmin, xmax=xmax,
                           ymin=ymin, ymax=ymax, ng=4)

        # first figure out the BCs
        xlb_type = self.rp.get_param("mesh.xlboundary")
        xrb_type = self.rp.get_param("mesh.xrboundary")
        ylb_type = self.rp.get_param("mesh.ylboundary")
        yrb_type = self.rp.get_param("mesh.yrboundary")

        bc_dens = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                                 ylb=ylb_type, yrb=yrb_type)

        # if we are reflecting, we need odd reflection in the normal
        # directions for the velocity
        bc_xodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="x")

        bc_yodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="y")

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
        for bc in [xlb_type, xrb_type, ylb_type, yrb_type]:
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
        self.base["D0"] = np.zeros((mygli.qy), dtype=np.float64)
        self.base["Dh0"] = np.zeros((myg.qy), dtype=np.float64)
        self.base["p0"] = np.zeros((myg.qy), dtype=np.float64)

        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.base, self.rp)')


        # add metric
        alpha = 1.
        beta = [0., 0., 0.]
        gamma = np.ones(3)
        self.metric = metric.Metric(self.cc_data, alpha, beta, gamma)

        # Construct zeta_0

        #gamma = self.rp.get_param("eos.gamma")
        #self.base["zeta"] = self.base["p0"]**(1.0/gamma)

        self.base["zeta"] = np.zeros((mygli.qy), dtype=np.float64)

        # we'll also need zeta_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        self.base["zeta-edges"] = np.zeros((myg.qy), dtype=np.float64)

        self.updateZeta(myg)




    def updateZeta(self, myg):
        """
        TODO: write this

        Parameters
        ----------
        myg : Grid2d object
            grid on which zeta lives
        """

        #find variables
        D0 = self.base["D0"]
        u0 = self.metric.calcu0

        # going to cheat with EoS and just say that p = (gamma-1)rho, so
        # 1/Gamma1*p = 1/rho*gamma = u^0/D*gamma
        # and
        # 1/Gamma1*p * dp/dr = d ln rho / dr = d ln (D/u^0) / dr
        #
        # So, if we integrate over this wrt r we get D/u^0, so zeta =
        # exp(D/u^0).

        zeta = np.exp(D0 / u0)

        self.base["zeta"] = zeta

        # we'll also need zeta_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        self.base["zeta-edges"][myg.jlo+1:myg.jhi+1] = \
            0.5*(self.base["zeta"][myg.jlo  :myg.jhi] +
                 self.base["zeta"][myg.jlo+1:myg.jhi+1])
        self.base["zeta-edges"][myg.jlo] = self.base["zeta"][myg.jlo]
        self.base["zeta-edges"][myg.jhi+1] = self.base["zeta"][myg.jhi]






    def make_prime(self, a, a0):
        return a - a0[np.newaxis,:]


    def timestep(self):
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
        if not np.max(np.abs(u)) == 0:
            xtmp = np.min(myg.dx/(np.abs(u[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])))
        if not np.max(np.abs(v)) == 0:
            ytmp = np.min(myg.dy/(np.abs(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])))

        dt = cfl*min(xtmp, ytmp)

        # We need an alternate timestep that accounts for buoyancy, to
        # handle the case where the velocity is initially zero.

        # FIXME: Need to do this relativistically!!!!!!!!!!!!
        D = self.cc_data.get_var("density")
        D0 = self.base["D0"]
        Dprime = self.make_prime(D, D0)

        g = self.rp.get_param("lm-atmosphere.grav")

        F_buoy = np.max(np.abs(Dprime[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*g)/
                        D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])

        dt_buoy = np.sqrt(2.0*myg.dx/F_buoy)

        dt = min(dt, dt_buoy)
        print("timestep is {}".format(dt))

        return dt


    def preevolve(self):
        """
        preevolve is called before we being the timestepping loop.  For
        the low Mach solver, this does an initial projection on the
        velocity field and then goes through the full evolution to get the
        value of phi.  The fluid state (D, u, v) is then reset to values
        before this evolve.
        """

        myg = self.cc_data.grid

        D = self.cc_data.get_var("density")
        #Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # the coefficent for the elliptic equation is zeta_0^2/D
        coeff = 1.0/D[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        zeta = self.base["zeta"]
        coeff = coeff*zeta[np.newaxis,myg.jlo-1:myg.jhi+2]**2

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

        # first compute div{zeta_0 U}
        div_zeta_U = mg.soln_grid.scratch_array()

        # u/v are cell-centered, divU is cell-centered
        div_zeta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*zeta[np.newaxis,mg.jlo:mg.jhi+1]* \
                (u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(zeta[np.newaxis,myg.jlo+1:myg.jhi+2]* \
                 v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 zeta[np.newaxis,myg.jlo-1:myg.jhi  ]*
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        # solve D (zeta_0^2/D) G (phi/zeta_0) = D( zeta_0 U )

        # set the RHS to div_zeta_U and solve
        mg.init_RHS(div_zeta_U)
        mg.solve(rtol=1.e-10)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.cc_data.get_var("phi")
        phi[:,:] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of phi and update the
        # velocities
        # FIXME: this update only needs to be done on the interior
        # cells -- not ghost cells
        gradp_x, gradp_y = mg.get_solution_gradient(grid=myg)

        coeff = 1.0/D[:,:]
        coeff = coeff*zeta[np.newaxis,:]

        u[:,:] -= coeff*gradp_x
        v[:,:] -= coeff*gradp_y

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 2. now get an approximation to gradp at n-1/2 by going through the
        # evolution.

        # store the current solution -- we'll restore it in a bit
        orig_data = patch.cell_center_data_clone(self.cc_data)

        # get the timestep
        dt = self.timestep()

        # evolve
        self.evolve(dt)

        # update gradp_x and gradp_y in our main data object
        new_gp_x = self.cc_data.get_var("gradp_x")
        new_gp_y = self.cc_data.get_var("gradp_y")

        orig_gp_x = orig_data.get_var("gradp_x")
        orig_gp_y = orig_data.get_var("gradp_y")

        orig_gp_x[:,:] = new_gp_x[:,:]
        orig_gp_y[:,:] = new_gp_y[:,:]

        self.cc_data = orig_data

        print("done with the pre-evolution")


    def evolve(self, dt):
        """
        Evolve the low Mach system through one timestep.

        FIXME: The base states are never evolved. This should definitely be
        rectified.
        """

        D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        gradp_x = self.cc_data.get_var("gradp_x")
        gradp_y = self.cc_data.get_var("gradp_y")


        # note: the base state quantities do not have valid ghost cells
        # need to update zeta as D0, u0 etc change every time step
        self.updateZeta(self.cc_data.grid)
        zeta = self.base["zeta"]
        zeta_edges = self.base["zeta-edges"]

        D0 = self.base["D0"]
        Dh0 = self.base["Dh0"]

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid


        #---------------------------------------------------------------------
        # create the limited slopes of D, u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("lm-atmosphere.limiter")
        if limiter == 0: limitFunc = reconstruction_f.nolimit
        elif limiter == 1: limitFunc = reconstruction_f.limit2
        else: limitFunc = reconstruction_f.limit4


        ldelta_rx = limitFunc(1, D, myg.qx, myg.qy, myg.ng)
        ldelta_ux = limitFunc(1, u, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v, myg.qx, myg.qy, myg.ng)

        ldelta_ry = limitFunc(2, D, myg.qx, myg.qy, myg.ng)
        ldelta_uy = limitFunc(2, u, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v, myg.qx, myg.qy, myg.ng)

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
        print("  making MAC velocities")

        # create the coefficient to the grad (pi/zeta) term
        # FIXME: Check this coefficient - think it may actually be zeta/Dh u^0
        coeff = self.aux_data.get_var("coeff")
        coeff[:,:] = 1.0/D[:,:]
        coeff[:,:] = coeff*zeta[np.newaxis,:]
        self.aux_data.fill_BC("coeff")

        # create the source term
        source = self.aux_data.get_var("source_y")

        g = self.rp.get_param("lm-atmosphere.grav")
        Dprime = self.make_prime(D, D0)

        # FIXME: Need to correct this to make relativistic
        source[:,:] = Dprime*g/D
        self.aux_data.fill_BC("source_y")

        u_MAC, v_MAC = lm_interface_f.mac_vels(myg.qx, myg.qy, myg.ng,
                                               myg.dx, myg.dy, dt,
                                               u, v,
                                               ldelta_ux, ldelta_vx,
                                               ldelta_uy, ldelta_vy,
                                               coeff*gradp_x, coeff*gradp_y,
                                               source)


        #---------------------------------------------------------------------
        # do a MAC projection to make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve D (zeta_0^2/D) G phi = D (zeta_0 U^MAC), where
        # phi is cell centered, and U^MAC is the MAC-type staggered
        # grid of the advective velocities.

        print("  MAC projection")

        # create the coefficient array: zeta**2/D
        coeff = 1.0/D[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        coeff = coeff*zeta[np.newaxis,myg.jlo-1:myg.jhi+2]**2

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

        # first compute div{zeta_0 U}
        div_zeta_U = mg.soln_grid.scratch_array()

        # MAC velocities are edge-centered.  div{zeta_0 U} is cell-centered.
        div_zeta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            zeta[np.newaxis,myg.jlo:myg.jhi+1]*(
                u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            (zeta_edges[np.newaxis,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             zeta_edges[np.newaxis,myg.jlo  :myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy

        # solve the Poisson problem
        mg.init_RHS(div_zeta_U)
        mg.solve(rtol=1.e-12)


        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/zeta_0
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC[:,:] = mg.get_solution(grid=myg)

        coeff = self.aux_data.get_var("coeff")
        coeff[:,:] = 1.0/D[:,:]
        coeff[:,:] = coeff*zeta[np.newaxis,:]
        self.aux_data.fill_BC("coeff")

        coeff_x = myg.scratch_array()
        coeff_x[myg.ilo-3:myg.ihi+2,myg.jlo:myg.jhi+1] = \
                0.5*(coeff[myg.ilo-2:myg.ihi+3,myg.jlo:myg.jhi+1] +
                     coeff[myg.ilo-3:myg.ihi+2,myg.jlo:myg.jhi+1])

        coeff_y = myg.scratch_array()
        coeff_y[myg.ilo:myg.ihi+1,myg.jlo-3:myg.jhi+2] = \
                0.5*(coeff[myg.ilo:myg.ihi+1,myg.jlo-2:myg.jhi+3] +
                     coeff[myg.ilo:myg.ihi+1,myg.jlo-3:myg.jhi+2])

        # we need the MAC velocities on all edges of the computational domain
        # here we do U = U - (zeta_0/D) grad (phi/zeta_0)
        u_MAC[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] -= \
                coeff_x[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1]* \
                (phi_MAC[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1] -
                 phi_MAC[myg.ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx

        v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] -= \
                coeff_y[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+2]* \
                (phi_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+2] -
                 phi_MAC[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1])/myg.dy


        #---------------------------------------------------------------------
        # predict D to the edges and do its conservative update
        #
        # Add source terms at start/end
        #---------------------------------------------------------------------
        D_xint, D_yint = lm_interface_f.D_states(myg.qx, myg.qy, myg.ng,
                                                       myg.dx, myg.dy, dt,
                                                       D, u_MAC, v_MAC,
                                                       ldelta_rx, ldelta_ry)

        D_old = D.copy()

        D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt*(
            #  (D u)_x
            (D_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             D_xint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx +
            #  (D v)_y
            (D_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             D_yint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy )

        self.cc_data.fill_BC("density")



        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        #
        # Exactly the same as for density
        #---------------------------------------------------------------------
        Dh_xint, Dh_yint = lm_interface_f.Dh_states(myg.qx, myg.qy, myg.ng,
                                                       myg.dx, myg.dy, dt,
                                                       Dh, u_MAC, v_MAC,
                                                       ldelta_rx, ldelta_ry)

        Dh_old = Dh.copy()

        Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt*(
            #  (Dh u)_x
            (Dh_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             Dh_xint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx +
            #  (Dh v)_y
            (Dh_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             Dh_yint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy )

        self.cc_data.fill_BC("enthalpy")

        #---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        print("  making u, v edge states")

        coeff = self.aux_data.get_var("coeff")
        coeff[:,:] = 2.0/(D[:,:] + D_old[:,:])

        # FIXME: check if need to recalculate zeta as D0 may have changed

        coeff[:,:] = coeff*zeta[np.newaxis,:]
        self.aux_data.fill_BC("coeff")

        u_xint, v_xint, u_yint, v_yint = \
               lm_interface_f.states(myg.qx, myg.qy, myg.ng,
                                     myg.dx, myg.dy, dt,
                                     u, v,
                                     ldelta_ux, ldelta_vx,
                                     ldelta_uy, ldelta_vy,
                                     coeff*gradp_x, coeff*gradp_y,
                                     source,
                                     u_MAC, v_MAC)


        #---------------------------------------------------------------------
        # update U to get the provisional velocity field
        #---------------------------------------------------------------------
        print("  doing provisional update of u, v")

        # compute (U.grad)U

        # we want u_MAC U_x + v_MAC U_y
        advect_x = myg.scratch_array()
        advect_y = myg.scratch_array()

        advect_x[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] +
                 u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]) * \
            (u_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             u_xint[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] +
                 v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]) * \
            (u_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             u_yint[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy

        advect_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] +
                 u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]) * \
            (v_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             v_xint[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] +
                 v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]) * \
            (v_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             v_yint[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy


        proj_type = self.rp.get_param("lm-atmosphere.proj_type")


        if proj_type == 1:
            u[:,:] -= (dt*advect_x[:,:] + dt*gradp_x[:,:])
            v[:,:] -= (dt*advect_y[:,:] + dt*gradp_y[:,:])

        elif proj_type == 2:
            u[:,:] -= dt*advect_x[:,:]
            v[:,:] -= dt*advect_y[:,:]


        # add the gravitational source

        # FIXME: Make relativistic !!!!!!!!!!
        D_half = 0.5*(D + D_old)
        Dprime = self.make_prime(D_half, D0)
        source = Dprime*g/D_half

        v[:,:] += dt*source

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        print("min/max D = {}, {}".format(np.min(D), np.max(D)))
        print("min/max u   = {}, {}".format(np.min(u), np.max(u)))
        print("min/max v   = {}, {}".format(np.min(v), np.max(v)))


        #---------------------------------------------------------------------
        # project the final velocity
        #---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)
        print("  final projection")

        # create the coefficient array: zeta**2/D
        # FIXME: probably need to recalculate zeta
        coeff = 1.0/D[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        coeff = coeff*zeta[np.newaxis,myg.jlo-1:myg.jhi+2]**2

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

        # first compute div{zeta_0 U}

        # u/v are cell-centered, divU is cell-centered
        div_zeta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*zeta[np.newaxis,mg.jlo:mg.jhi+1]* \
                (u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(zeta[np.newaxis,myg.jlo+1:myg.jhi+2]* \
                 v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 zeta[np.newaxis,myg.jlo-1:myg.jhi  ]*
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        mg.init_RHS(div_zeta_U/dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess[mg.ilo-1:mg.ihi+2,mg.jlo-1:mg.jhi+2] = \
           phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        mg.init_solution(phiGuess)

        # solve
        mg.solve(rtol=1.e-12)


        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi[:,:] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)


        # U = U - (zeta_0/D) grad (phi/zeta_0)
        coeff = 1.0/D[:,:]
        coeff = coeff*zeta[np.newaxis,:]

        u[:,:] -= dt*coeff*gradphi_x
        v[:,:] -= dt*coeff*gradphi_y

        # store gradp for the next step

        if proj_type == 1:
            gradp_x[:,:] += gradphi_x[:,:]
            gradp_y[:,:] += gradphi_y[:,:]

        elif proj_type == 2:
            gradp_x[:,:] = gradphi_x[:,:]
            gradp_y[:,:] = gradphi_y[:,:]

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        self.cc_data.fill_BC("gradp_x")
        self.cc_data.fill_BC("gradp_y")


    def dovis(self):
        """
        Do runtime visualization
        """
        plt.clf()

        #plt.rc("font", size=10)

        D = self.cc_data.get_var("density")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        myg = self.cc_data.grid

        magvel = np.sqrt(u**2 + v**2)

        vort = myg.scratch_array()

        dv = 0.5*(v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                  v[myg.ilo-1:myg.ihi,  myg.jlo:myg.jhi+1])/myg.dx

        du = 0.5*(u[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                  u[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        # for some reason, setting vort here causes the density in the
        # simulation to NaN.  Seems like a bug (in python?)
        #vort[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = dv - du

        fig, axes = plt.subplots(nrows=1, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.25)

        fields = [D, magvel]
        field_names = [r"$\D$", r"|U|"] #, r"$\nabla \times U$", r"$\D$"]

        for n in range(len(fields)):
            ax = axes.flat[n]

            f = fields[n]

            img = ax.imshow(np.transpose(f[myg.ilo:myg.ihi+1,
                                           myg.jlo:myg.jhi+1]),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(field_names[n])

            #plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()


    def finalize(self):
        """
        Do any final clean-ups for the simulation and call the problem's
        finalize() method.
        """
        exec(self.problem_name + '.finalize()')
