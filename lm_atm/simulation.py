from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
#import lm_atm.problems.bubble as bubble

from lm_atm.problems import *
#import lm_atm.LM_atm_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
import multigrid.variable_coeff_MG as vcMG
import metric
from util import profile
import lm_atm_interface as lm_int

"""
TODO: Compare to MATLAB code to see if I did anything else

TODO: Do some dimensional analysis or similar to work out initial conditions?

FIXME: Make sure ghost cells in both full state and base state are updated.

FIXME: After ~15 timesteps, the elliptic equation solver diverges.
       Need to work out why this happens and how to stop it.
       Coincides with density/enthalpy becoming negative which is definitely
       not good.
"""

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
        self.base_data = None
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
        myg1d = patch.Grid1d(ny, ymin=ymin, ymax=ymax, ng=4)

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
        my_data.register_var("tracer", bc_dens)

        my_data.create()

        self.cc_data = my_data


        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = patch.CellCenterData2d(myg)

        aux_data.register_var("coeff", bc_dens)
        aux_data.register_var("source_y", bc_yodd)
        aux_data.register_var("S", bc_yodd)

        aux_data.create()
        self.aux_data = aux_data


        # we also need storage for the 1-d base state -- we'll store this
        # in the main class directly.
        base_data = patch.CellCenterData1d(myg1d)
        base_data.register_var("D0", bc_dens)
        base_data.register_var("Dh0", bc_dens)
        base_data.register_var("p0", bc_dens)


        # add metric
        g = self.rp.get_param("lm-atmosphere.grav")
        c = self.rp.get_param("lm-atmosphere.c")
        alpha = np.sqrt(c**2 * np.ones(myg.qy) + 2. * g / myg.y)
        beta = [0., 0., 0.] #really flat
        gamma = np.eye(3) #extremely flat
        self.metric = metric.Metric(self.cc_data, self.rp, alpha, beta, gamma)

        # Construct zeta
        base_data.register_var("zeta", bc_dens)

        # we'll also need zeta on vertical edges -- on the domain edges,
        # just do piecewise constant
        base_data.register_var("zeta-edges", bc_dens)

        base_data.create()
        self.base_data = base_data


        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.base_data, self.rp)')

        self.updateZeta(myg)



    @staticmethod
    def lateralAvg(a):
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





    def updateZeta(self, myg):
        """
        Update zeta in the interior and on the edges. Assumes all other
        variables are up to date.

        Parameters
        ----------
        myg : Grid2d object
            grid on which zeta lives
        """

        #find variables
        D0 = self.base_data.get_var("D0")
        u0 = self.metric.calcu0()

        # FIXME: don't cheat the EoS
        # going to cheat with EoS and just say that p = rho^gamma, so
        # 1/Gamma1*p = 1/Gamma1*rho^gamma = u0^gamma/gamma*D^gamma
        # and
        # 1/Gamma1*p * dp/dr = d ln rho / dr = d ln (D/u^0) / dr
        #
        # So, if we integrate over this wrt r we get D/u^0, so zeta =
        # exp(D/u^0).
        zeta = self.base_data.get_var("zeta")
        zeta_edges = self.base_data.get_var("zeta-edges")
        # do some u0 averaging?
        zeta[myg.jlo:myg.jhi+1] = np.exp(D0[myg.jlo:myg.jhi+1] / \
            self.lateralAvg(u0)[myg.jlo:myg.jhi+1])

        self.base_data.fill_BC("zeta")


        # we'll also need zeta_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        zeta_edges[myg.jlo+1:myg.jhi+1] = \
            0.5*(zeta[myg.jlo  :myg.jhi] +
                 zeta[myg.jlo+1:myg.jhi+1])
        zeta_edges[myg.jlo] = zeta[myg.jlo]
        zeta_edges[myg.jhi+1] = zeta[myg.jhi]

        self.base_data.fill_BC("zeta-edges")


    def calcConstraint(self, zeta):
        """
        Calculates and returns the source terms in the constraint:
        zeta (S - dpdt / Gamma1 p)

        Parameters
        ----------
        zeta : float array
            zeta

        Returns
        -------
        calcConstraint : float array
            zeta (S - dpdt / Gamma1 p)
        """

        #S = -Gamma^mu_{mu nu}U^nu. In easy metric, this reduces to
        #Gamma^t_tr U^r
        g = self.rp.get_param("lm-atmosphere.grav")
        #Gamma1 = gamma for our EoS.
        gamma = self.rp.get_param("eos.gamma")
        v = self.cc_data.get_var("y-velocity")

        S = self.aux_data.get_var("S")

        S[:,:] = g * v[:,:] / (self.cc_data.grid.y[np.newaxis,:] * self.metric.alpha[np.newaxis,:])**2
        self.aux_data.fill_BC("S")

        p0 = self.base_data.get_var("p0")

        # laterally averaged the source as p0 is x-independent
        dp0dt = self.lateralAvg(S[:,:]) * 0. #placeholder for now
        dp0dt[:] /= gamma * p0[:]

        return zeta[np.newaxis,:] * (S[:,:] - dp0dt[np.newaxis,:])


    @staticmethod
    def make_prime(a, a0):
        """
        Subtracts the base part of a state, a0, away from the full state, a, to
        give the perturbed state, a'.
        """
        return a - a0[np.newaxis,:]


    def timestep(self):
        """
        The timestep() function computes the advective timestep
        (CFL) constraint.  The CFL constraint says that information
        cannot propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the CFL
        step we actually take.

        Returns
        -------
        dt : float
            timestep
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

        dt = cfl * min(xtmp, ytmp)

        # We need an alternate timestep that accounts for buoyancy, to
        # handle the case where the velocity is initially zero.

        # FIXME: Need to do this relativistically!!!!!!!!!!!!
        D = self.cc_data.get_var("density")
        D0 = self.base_data.get_var("D0")
        Dprime = self.make_prime(D, D0)

        g = self.rp.get_param("lm-atmosphere.grav")

        #F_buoy = np.max(np.abs(Dprime[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*g)/
        #                D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])
        F_buoy = np.max([np.max(np.abs(g/myg.y[:]**2)), 1.e-20])

        dt_buoy = np.sqrt(2.0*myg.dy/F_buoy)

        dt = min(dt, dt_buoy)
        #dt = min(dt, 0.25)
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

        #D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")
        self.cc_data.fill_BC("enthalpy")


        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # the coefficent for the elliptic equation is zeta^2/Dhu0
        # haven't evolved anything yet don't need to update zeta
        u0 = self.metric.calcu0()

        #shall ones out Dh to stop divide by 0ness
        #Dh = np.ones(np.shape(Dh))
        coeff = 1.0/(Dh[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] * \
            u0[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2])
        zeta = self.base_data.get_var("zeta")
        coeff[:,:] *= zeta[np.newaxis,myg.jlo-1:myg.jhi+2]**2

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
        div_zeta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*zeta[np.newaxis,mg.jlo:mg.jhi+1]* \
                (u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(zeta[np.newaxis,myg.jlo+1:myg.jhi+2]* \
                 v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 zeta[np.newaxis,myg.jlo-1:myg.jhi  ]*
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        # solve
        # D(zeta^2/Dhu0) G (phi/zeta) + zeta(S - dp/dt / Gamma1*p)= D( zeta U )

        # set the RHS to div_zeta_U - zeta(S -...) and solve
        p0 = self.base_data.get_var("p0")
        constraint = self.calcConstraint(zeta)
        mg.init_RHS(div_zeta_U[:,:] - constraint[myg.ilo-1:myg.ihi+2, myg.jlo-1:myg.jhi+2])
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

        coeff = 1.0/(Dh[:,:] * u0[:,:])
        coeff[:,:] *= zeta[np.newaxis,:]

        u[:,:] -= coeff[:,:] * gradp_x[:,:]
        v[:,:] -= coeff[:,:] * gradp_y[:,:]

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")
        self.cc_data.fill_BC("enthalpy")


        # 2. now get an approximation to gradp at n-1/2 by going through the
        # evolution.

        # store the current solution -- we'll restore it in a bit
        orig_data = patch.cell_center_data_clone(self.cc_data)
        orig_base = patch.cell_center_data1d_clone(self.base_data)

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
        self.base_data = orig_base

        print("done with the pre-evolution")


    def evolve(self, dt):
        """
        Evolve the low Mach system through one timestep.

        Parameters
        ----------
        dt : float
            timestep
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
        zeta = self.base_data.get_var("zeta")
        zeta_edges = self.base_data.get_var("zeta-edges")

        D0 = self.base_data.get_var("D0")
        Dh0 = self.base_data.get_var("Dh0")

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid


        #---------------------------------------------------------------------
        # create the limited slopes of D, Dh, u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("lm-atmosphere.limiter")
        if limiter == 0: limitFunc = reconstruction_f.nolimit
        elif limiter == 1: limitFunc = reconstruction_f.limit2
        else: limitFunc = reconstruction_f.limit4

        # resize so that Fortran is happy
        D02d = np.array([D0,] * np.size(D0))
        Dh02d = np.array([Dh0,] * np.size(Dh0))

        # x slopes of r0, e0 surely just 0?
        ldelta_rx = limitFunc(1, D, myg.qx, myg.qy, myg.ng)
        ldelta_ex = limitFunc(1, Dh, myg.qx, myg.qy, myg.ng)
        ldelta_r0x = limitFunc(1, D02d, myg.qx, myg.qy, myg.ng)
        ldelta_e0x = limitFunc(1, Dh02d, myg.qx, myg.qy, myg.ng)
        ldelta_ux = limitFunc(1, u, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v, myg.qx, myg.qy, myg.ng)

        ldelta_ry = limitFunc(2, D, myg.qx, myg.qy, myg.ng)
        ldelta_ey = limitFunc(2, Dh, myg.qx, myg.qy, myg.ng)
        ldelta_r0y = limitFunc(2, D02d, myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh02d, myg.qx, myg.qy, myg.ng)
        ldelta_uy = limitFunc(2, u, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v, myg.qx, myg.qy, myg.ng)


        #---------------------------------------------------------------------
        # React full state through first half timestep
        # Assume for simplicity here that only non-zero Christoffel of
        # Gamma^mu_{mu nu} form is Gamma^t_{t r} = - g/r^2(1+2g/r) =
        # -g/r^2alpha^2
        #
        # This runs ReactState and basically calculates the sourcing.
        # Shall also react the base state as this has sourcing too.
        #
        # CHANGED: have attempted to include pressure term in energy eq.
        #---------------------------------------------------------------------
        g = self.rp.get_param("lm-atmosphere.grav")
        r = np.array([myg.y,] * myg.qx)

        christfl = -g / (r[:,:]**2 * (1.+ 2. * g / r[:,:]))

        D0[myg.jlo:myg.jhi+1] -= dt * 0.5 * \
            self.lateralAvg(D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] )

        Dh0[myg.jlo:myg.jhi+1] -= dt * 0.5 * \
            self.lateralAvg(Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]  + \
            dt * v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * g / \
            (r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2 * \
            self.metric.alpha[np.newaxis,myg.jlo:myg.jhi+1]**2) )

        D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt * \
            D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * 0.5 * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]
        Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt * \
            Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * 0.5 * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + \
            dt * v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * g / \
            (r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2 * \
            self.metric.alpha[np.newaxis,myg.jlo:myg.jhi+1]**2)


        #fill ghostcells
        self.cc_data.fill_BC("enthalpy")
        self.cc_data.fill_BC("density")
        self.base_data.fill_BC("D0")
        self.base_data.fill_BC("Dh0")


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
        #print("  making MAC velocities")

        # create the coefficient to the grad (pi/zeta) term
        u0 = self.metric.calcu0()
        coeff = self.aux_data.get_var("coeff")
        tracer = self.cc_data.get_var("tracer")

        coeff[:,:] = 1.0/(Dh[:,:] * u0[:,:])
        coeff[:,:] *= zeta[np.newaxis,:]
        self.aux_data.fill_BC("coeff")

        # create the source term
        source = self.aux_data.get_var("source_y")
        """
        #g = self.rp.get_param("lm-atmosphere.grav")
        Dprime = self.make_prime(D, D0)

        source[:,:] = Dprime*g/D
        """

        """
        Have attempted to do this relativistically.

        Source here is given by -U_j Dlnu0/Dt + Gamma_{rho nu j} U^nu U^rho.
        The second term in our simple time-lagged metric is just g/r^2.
        The first term presents some difficulty so shall try to ignore it for
        now.
        """
        source[:,:] = g  / r[:,:]**2
        self.aux_data.fill_BC("source_y")

        u_MAC, v_MAC = lm_int.mac_vels(myg, dt, u, v, ldelta_ux, ldelta_vx,
                                               ldelta_uy, ldelta_vy,
                                               coeff*gradp_x, coeff*gradp_y,
                                               source)


        #---------------------------------------------------------------------
        # do a MAC projection to make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve D (zeta^2/D) G phi = D (zeta U^MAC), where
        # phi is cell centered, and U^MAC is the MAC-type staggered
        # grid of the advective velocities.

        #print("  MAC projection")

        # create the coefficient array: zeta**2/Dhu0
        u0 = self.metric.calcu0()
        coeff = 1.0/(Dh[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] * \
            u0[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2])
        self.updateZeta(self.cc_data.grid)
        zeta = self.base_data.get_var("zeta")
        coeff[:,:] *= zeta[np.newaxis,myg.jlo-1:myg.jhi+2]**2


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
        div_zeta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            zeta[np.newaxis,myg.jlo:myg.jhi+1]*(
                u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            (zeta_edges[np.newaxis,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             zeta_edges[np.newaxis,myg.jlo  :myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy

        # solve the Poisson problem
        constraint = self.calcConstraint(zeta)
        mg.init_RHS(div_zeta_U[:,:] - constraint[myg.ilo-1:myg.ihi+2, myg.jlo-1:myg.jhi+2])
        #mg.solve(rtol=1.e-12)
        mg.solve(1.e-9)


        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/zeta
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC[:,:] = mg.get_solution(grid=myg)

        coeff = self.aux_data.get_var("coeff")
        u0 = self.metric.calcu0()
        coeff[:,:] = 1.0/(Dh[:,:] * u0[:,:])
        coeff[:,:] *= zeta[np.newaxis,:]
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
        # here we do U = U - (zeta/D) grad (phi/zeta)
        u_MAC[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] -= \
                coeff_x[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1]* \
                (phi_MAC[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1] -
                 phi_MAC[myg.ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx

        v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] -= \
                coeff_y[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+2]* \
                (phi_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+2] -
                 phi_MAC[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1])/myg.dy


        # resize so that Fortran is happy
        D02d = np.array([D0,] * np.size(D0))
        Dh02d = np.array([Dh0,] * np.size(Dh0))


        #---------------------------------------------------------------------
        # predict D to the edges and do its conservative update
        #---------------------------------------------------------------------
        D_xint, D_yint = lm_int.D_states(myg, dt, D, u_MAC, v_MAC, ldelta_rx,
                            ldelta_ry)

        D0_xint, D0_yint = lm_int.D_states(myg, dt, D02d, u_MAC,
                            v_MAC, ldelta_r0x, ldelta_r0y)

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


        #need to do some averaging as D0 is only 1d

        D02d[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt*(
            #  (D u)_x
            (D0_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             D0_xint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx +
            #  (D v)_y
            (D0_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             D0_yint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy )

        D0[:] = self.lateralAvg(D02d)


        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        #
        # Exactly the same as for density
        #---------------------------------------------------------------------
        Dh_xint, Dh_yint = lm_int.D_states(myg, dt, D, u_MAC, v_MAC, ldelta_ex,
                            ldelta_ey)

        Dh0_xint, Dh0_yint = lm_int.D_states(myg, dt, Dh02d, u_MAC,
                            v_MAC, ldelta_e0x, ldelta_e0y)

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

        # average laterally as Dh0 is 1d

        Dh02d[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt*(
            #  (Dh u)_x
            (Dh0_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             Dh0_xint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx +
            #  (Dh v)_y
            (Dh0_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             Dh0_yint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy )

        Dh0[:] = self.lateralAvg(Dh02d)

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")
        self.base_data.fill_BC("D0")
        self.base_data.fill_BC("Dh0")

        #---------------------------------------------------------------------
        # Enforce relativistic hydrostatic equilibrium on base pressure
        #
        # This is based on EnforceHSE.
        #---------------------------------------------------------------------

        u0 = self.metric.calcu0()

        # flatten u0 to make 1d for next equation.
        u0flat = self.lateralAvg(u0[:,:])
        p0 = self.base_data.get_var("p0")
        p0[1:] -= myg.dy * g * (Dh0[1:]/(u0flat[1:] * self.metric.alpha[1:]) +\
            Dh0[:-1]/(u0flat[:-1] * self.metric.alpha[:-1]))

        #p0[myg.jlo:myg.jhi+1] -= myg.dy * \
        #    (Dh0[myg.jlo+1:myg.jhi+2]*g /  (u0flat[myg.jlo+1:myg.jhi+2] * \
        #     self.metric.alpha[myg.jlo+1:myg.jhi+2]**2) + \
        #     Dh0[myg.jlo:myg.jhi+1]*g / (u0flat[myg.jlo:myg.jhi+1] * \
        #     self.metric.alpha[myg.jlo:myg.jhi+1]**2)) / 2.

        #self.base_data.fill_BC("p0")

        #---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------

        coeff = self.aux_data.get_var("coeff")
        coeff[:,:] = 2.0/((Dh[:,:] + Dh_old[:,:]) * u0[:,:])

        # Might not need to recalculate zeta but shall just in case
        self.updateZeta(self.cc_data.grid)
        zeta = self.base_data.get_var("zeta")
        coeff[:,:] *= zeta[np.newaxis,:]
        self.aux_data.fill_BC("coeff")

        u_xint, v_xint, u_yint, v_yint = \
            lm_int.states(myg, dt, u, v, ldelta_ux, ldelta_vx,
                                 ldelta_uy, ldelta_vy,
                                 coeff*gradp_x, coeff*gradp_y,
                                 source,
                                 u_MAC, v_MAC)


        #---------------------------------------------------------------------
        # update U to get the provisional velocity field
        #---------------------------------------------------------------------
        #print("  doing provisional update of u, v")

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

        # add the gravitational source (and pressure source)

        # CHANGED: Have added pressure terms to this.
        source[:,:] = g / myg.y[np.newaxis,:]**2 + v[:,:]**2 * g / \
            (myg.y[np.newaxis,:] * self.metric.alpha[np.newaxis,:])**2

        u[:,:] += dt * u[:,:] * v[:,:] * g / \
            (myg.y[np.newaxis,:] * self.metric.alpha[np.newaxis,:])**2
        v[:,:] += dt * source[:,:]
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        print("min/max D = {}, {}".format(np.min(D), np.max(D)))
        print("min/max Dh = {}, {}".format(np.min(Dh), np.max(Dh)))
        print("min/max u   = {}, {}".format(np.min(u), np.max(u)))
        print("min/max v   = {}, {}".format(np.min(v), np.max(v)))

        #calculate the sound speed
        cs = myg.scratch_array()
        gamma = self.rp.get_param("eos.gamma")
        cs[:,:] = gamma * (gamma - 1.) / (2. - gamma)
        cs[:,:] *= (D[:,:] + Dh[:,:]) / D[:,:]
        cs[:,:] = np.sqrt(np.abs(cs[:,:]))

        print("min/max c_s   = {}, {}".format(np.min(cs), np.max(cs)))

        #calculate and print Mach number
        speed = np.sqrt(u[:,:]**2 + v[:,:]**2)
        M = speed[:,:]/cs[:,:]
        print("min/max M   = {}, {}".format(np.min(M), np.max(M)))

        #---------------------------------------------------------------------
        # React full and base state through second half timestep
        # Assume for simplicity here that only non-zero Christoffel of
        # Gamma^mu_{mu nu} form is Gamma^t_{t r} = g/(1+2gr)
        #
        # This runs ReactState.
        #---------------------------------------------------------------------

        D0[myg.jlo:myg.jhi+1] -= dt * 0.5 * \
            self.lateralAvg(D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] )

        #CHANGED: included pressure terms

        Dh0[myg.jlo:myg.jhi+1] -= dt * 0.5 * \
            self.lateralAvg(Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + \
            dt * v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * g / \
            (r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2 * \
            self.metric.alpha[np.newaxis,myg.jlo:myg.jhi+1]**2) )

        D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt * \
            D[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * 0.5 * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]
        Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt * \
            Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * 0.5 * \
            christfl[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + \
            dt * v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * \
            Dh[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] * g / \
            (r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2 * \
            self.metric.alpha[np.newaxis,myg.jlo:myg.jhi+1]**2)

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")
        self.base_data.fill_BC("D0")
        self.base_data.fill_BC("Dh0")



        #---------------------------------------------------------------------
        # project the final velocity
        #---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)
        print("  final projection")

        # create the coefficient array: zeta**2/Dhu0
        u0 = self.metric.calcu0()
        coeff = 1.0/(Dh[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] * \
            u0[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2])
        self.updateZeta(self.cc_data.grid)
        zeta = self.base_data.get_var("zeta")
        coeff[:,:] *= zeta[np.newaxis,myg.jlo-1:myg.jhi+2]**2


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
                                 verbose=0, nsmooth=20)

        # first compute div{zeta U}

        # u/v are cell-centered, divU is cell-centered
        div_zeta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*zeta[np.newaxis,mg.jlo:mg.jhi+1]* \
                (u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(zeta[np.newaxis,myg.jlo+1:myg.jhi+2]* \
                 v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 zeta[np.newaxis,myg.jlo-1:myg.jhi  ]*
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        constraint = self.calcConstraint(zeta)
        mg.init_RHS(div_zeta_U[:,:]/dt - constraint[myg.ilo-1:myg.ihi+2, myg.jlo-1:myg.jhi+2]/dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess[mg.ilo-1:mg.ihi+2,mg.jlo-1:mg.jhi+2] = \
           phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        #mg.init_solution(phiGuess)

        # solve
        #mg.solve(rtol=1.e-12)
        # FIXME: the error in this diverges after a few timesteps
        mg.solve(rtol=1.e-9)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi[:,:] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)


        # U = U - (zeta/Dhu0) grad (phi/zeta)
        u0 = self.metric.calcu0()
        coeff = 1.0/(Dh[:,:] * u0[:,:])
        coeff[:,:] *= zeta[np.newaxis,:]
        #self.aux_data.fill_BC("coeff")

        u[:,:] -= dt * coeff[:,:] * gradphi_x[:,:]
        v[:,:] -= dt * coeff[:,:] * gradphi_y[:,:]

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

        #zeta2d = np.array([zeta,] * np.size(zeta))

        tracer[:,:] = v.copy()
        print(Dh[30, myg.jhi-3:])




    def dovis(self):
        """
        Do runtime visualization
        """
        plt.clf()

        #plt.rc("font", size=10)
        D = self.cc_data.get_var("density")
        Dh= self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        #tracer = self.cc_data.get_var("tracer")
        #print('zeta size', np.shape(zeta))

        myg = self.cc_data.grid

        magvel = np.sqrt(u**2 + v**2)

        cs = myg.scratch_array()
        #gamma = self.rp.get_param("eos.gamma")
        gamma = 1.4
        cs[:,:] = gamma * (gamma - 1.) / (2. - gamma)
        cs[:,:] *= (D[:,:] + Dh[:,:]) / D[:,:]
        cs[:,:] = np.sqrt(np.abs(cs[:,:]))
        M = magvel[:,:]/cs[:,:]

        vort = myg.scratch_array()

        dv = 0.5*(v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                  v[myg.ilo-1:myg.ihi,  myg.jlo:myg.jhi+1])/myg.dx

        du = 0.5*(u[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                  u[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        # for some reason, setting vort here causes the density in the
        # simulation to NaN.  Seems like a bug (in python?)
        vort[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = dv - du

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.25)


        #fields = [D, magvel]
        fields = [D, magvel, u, v]
        #fields = [D, tracer]
        field_names = [r"$D$", r"$|U|$", r"$u$", r"$v$"] #, r"$\nabla \times U$", r"$D$"]

        for n in range(len(fields)):

            ax = axes.flat[n]

            f = fields[n]

            if n < 2:
                cmap = plt.cm.jet
                vmin = np.min(f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])
                vmax = np.max(f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])
            else:
                cmap = plt.cm.seismic
                vmin = -np.max(np.abs(f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]))
                vmax = +np.max(np.abs(f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]))

            img = ax.imshow(np.transpose(f[myg.ilo:myg.ihi+1,
                                           myg.jlo:myg.jhi+1]),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap = cmap, vmin = vmin, vmax = vmax)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(field_names[n])

            plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)
        #plt.tight_layout()

        plt.draw()


    def finalize(self):
        """
        Do any final clean-ups for the simulation and call the problem's
        finalize() method.
        """
        exec(self.problem_name + '.finalize()')
