from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from lm_atm.problems import *
import mesh.patch as patch
import multigrid.variable_coeff_MG as vcMG
import metric
import sys
from util import profile
from util import msg
import lm_atm_interface as lm_int
import lm_atm.LM_atm_interface_f as lm_interface_f
import mesh.reconstruction as reconstruction_f
from simulation_null import NullSimulation, grid_setup, bc_setup

"""
CHANGED:  Added in vanilla pyro's new null initialisation

TODO:  Break code down more into separate functions

TODO:  Do some dimensional analysis or similar to work out initial conditions?

FIXME: Make sure ghost cells in both full state and base state are updated.

FIXME: After ~15 timesteps, the elliptic equation solver diverges.
       Need to work out why this happens and how to stop it.
       Coincides with density/enthalpy becoming negative which is definitely
       not good.

TODO:  Implement some abstract base classes?
       Could create some abstract base classes to help create problems and
       set up initial conditions

TODO:  Identify where code normally breaks and put in validation

FIXME: There is no p0 evolution - perhaps derive from D0 then enforce HSE?
"""

class Basestate(object):
    def __init__(self, ny, ng=0):
        self.ny = ny
        self.ng = ng
        self.qy = ny + 2*ng

        self.d = np.zeros((self.qy), dtype=np.float64)

        self.jlo = ng
        self.jhi = ng+ny-1

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

        NullSimulation.__init__(self, solver_name, problem_name, rp, timers=timers)
        self.aux_data = None
        self.base_data = None
        self.metric = None

    def initialize(self):
        """
        Initialize the grid and variables for low Mach atmospheric flow
        and set the initial conditions for the chosen problem.
        """

        # setup the grid
        myg = grid_setup(self.rp, ng=4)
        myg1d = patch.Grid1d(myg.ny, ymin=myg.ymin, ymax=myg.ymax, ng=4)
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
        # currently 2 spatial dimensions: beta is 2-vector, gamma 2x2 matrix
        g = self.rp.get_param("lm-atmosphere.grav")
        c = self.rp.get_param("lm-atmosphere.c")
        R = self.rp.get_param("lm-atmosphere.radius")
        alpha = np.sqrt(1. - 2. * g * (1. - myg.y[:]/R)/ c**2)
        beta = [0., 0.] #really flat
        gamma = np.eye(2) #extremely flat
        self.metric = metric.Metric(self.cc_data, self.rp, alpha, beta, gamma)

        # Construct xi and zeta
        base_data.register_var("xi", bc_dens)
        base_data.register_var("zeta", bc_dens)

        # we'll also need xi and zeta on vertical edges -- on the domain edges,
        # just do piecewise constant
        base_data.register_var("xi-edges", bc_dens)
        base_data.register_var("zeta-edges", bc_dens)

        base_data.create()
        self.base_data = base_data


        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.base_data, self.rp, self.metric)')

        self.updateXi(myg)
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


    @staticmethod
    def checkXSymmetry(grid, nx):
        """
        Checks to see if a grid is symmetric in the x-direction.

        Parameters
        ----------
        grid : float array
            2d grid to be checked
        nx :
            grid x-dimension

        Returns
        -------
        sym : boolean
            whether or not the grid is symmetric
        """

        halfGrid = np.abs(grid[-np.floor(nx/2.):, :]) - \
            np.abs(grid[np.floor(nx/2.)-1::-1, :])

        if np.max(np.abs(halfGrid)) > 1.e-14:
            print('Oh no! An asymmetry has occured!')
            print('Asymmetry has amplitude: ', np.max(np.abs(halfGrid)))



    def updateXi(self, myg):
        """
        Update xi in the interior and on the edges. Assumes all other
        variables are up to date.

        Parameters
        ----------
        myg : Grid2d object
            grid on which xi lives
        """

        #find variables
        p0 = self.base_data.get_var("p0")
        gamma = self.rp.get_param("eos.gamma")
        # xi = exp(ln p0 / Gamma1)
        xi = self.base_data.get_var("xi")
        xi_edges = self.base_data.get_var("xi-edges")

        try:
            xi.v()[:] = np.exp(np.log(p0.v()) / gamma)
        except FloatingPointError:
            print(np.min(p0.v()))

        self.base_data.fill_BC("xi")


        # we'll also need xi_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        xi_edges.jp(1)[:] = 0.5 * (xi.v() + xi.jp(1))
        xi_edges.d[myg.jlo] = xi.d[myg.jlo]
        xi_edges.d[myg.jhi+1] = xi.d[myg.jhi]

        self.base_data.fill_BC("xi-edges")


    def updateZeta(self, myg):
        """
        Update zeta in the interior and on the edges. Assumes all other
        variables are up to date.

        Parameters
        ----------
        myg : Grid2d object
        grid on which xi lives
        """

        #find variables
        D0 = self.base_data.get_var("D0")
        u0 = self.metric.calcu0()
        # going to cheat with EoS and just say that p = rho^gamma, so
        # 1/Gamma1*p = 1/Gamma1*rho^gamma = u0^gamma/gamma*D^gamma
        # and
        # 1/Gamma1*p * dp/dr = d ln rho / dr = d ln (D/u^0) / dr
        #
        # So, if we integrate over this wrt r we get ln(D/u^0), so zeta =
        # exp(ln(D/u^0)) = D/u^0
        zeta = self.base_data.get_var("zeta")
        zeta_edges = self.base_data.get_var("zeta-edges")

        try:
            zeta.v()[:] = D0.v() / self.lateralAvg(u0.v())
        except FloatingPointError:
            print('D0: ', np.max(D0.v()))
            print('u0: ', np.max(u0.v()))
        self.base_data.fill_BC("zeta")

        # we'll also need zeta_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        zeta_edges.jp(1)[:] = 0.5*(zeta.v() + zeta.jp(1))
        zeta_edges.d[myg.jlo] = zeta.d[myg.jlo]
        zeta_edges.d[myg.jhi+1] = zeta.d[myg.jhi]
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
        c = self.rp.get_param("lm-atmosphere.c")
        R = self.rp.get_param("lm-atmosphere.radius")
        gamma = self.rp.get_param("eos.gamma")
        v = self.cc_data.get_var("y-velocity")

        S = self.aux_data.get_var("S")

        S.d[:,:] = -g * v.d / (c**2 * R * \
                 self.metric.alpha.d**2)
        self.aux_data.fill_BC("S")

        p0 = self.base_data.get_var("p0")

        # laterally averaged the source as p0 is x-independent
        dp0dt = self.lateralAvg(S.d) * 0. #placeholder for now
        dp0dt[:] /= gamma * p0.d

        return zeta.v2d(buf=zeta.ng) * (S.d - dp0dt[np.newaxis,:])


    @staticmethod
    def make_prime(a, a0):
        """
        Subtracts the base part of a state, a0, away from the full state, a, to
        give the perturbed state, a'.
        """
        return a - a0.v2d(buf=a0.ng)


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
        #gamma = self.rp.get_param("eos.gamma")#####
        #u0 = self.metric.calcu0()#####

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        #D = self.cc_data.get_var("density")


        # the timestep is min(dx/|u|, dy|v|)
        xtmp = ytmp = 100.
        #cs = gamma * (D[:,:]/u0[:,:])**(gamma - 1.)
        #cs[:,:] = np.sqrt(np.abs(cs[:,:]))
        #print('cs shape: ', np.shape(cs))
        if not np.max(np.abs(u.d)) < 1.e-5:
            xtmp = \
                np.min(myg.dx/np.abs(u.v()))
        if not np.max(np.abs(v.d)) < 1.e-5:
            ytmp = \
                np.min(myg.dy/np.abs(v.v()))

        dt = cfl * min(xtmp, ytmp)

        # We need an alternate timestep that accounts for buoyancy to
        # handle the case where the velocity is initially zero.
        g = self.rp.get_param("lm-atmosphere.grav")
        R = self.rp.get_param("lm-atmosphere.radius")
        c = self.rp.get_param("lm-atmosphere.c")

        F_buoy = np.max([np.max(g / (R * c**2)), 1.e-20])
        dt_buoy = np.sqrt(2.0*myg.dy/F_buoy)

        dt = min(dt, dt_buoy)

        return dt


    def ReactState(self, dt):
        """
        React full and base states through half timestep
        Assume for simplicity here that only non-zero Christoffel of
        Gamma^mu_{mu nu} form is Gamma^t_{t r} = - g/r^2(1+2g/r) =
        -g/r^2alpha^2

        This runs ReactState and basically calculates the sourcing.
        Shall also react the base state as this has sourcing too.

        CHANGED: have attempted to include pressure term in energy eq.
        """
        g = self.rp.get_param("lm-atmosphere.grav")
        c = self.rp.get_param("lm-atmosphere.c")
        R = self.rp.get_param("lm-atmosphere.radius")
        #myg = self.cc_data.grid
        D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        v = self.cc_data.get_var("y-velocity")
        D0 = self.base_data.get_var("D0")
        Dh0 = self.base_data.get_var("Dh0")

        christfl = self.base_data.grid.scratch_array()

        christfl.d[:] = g/(self.metric.alpha.d**2 * c**2 * R)

        D0.v()[:] -= dt * 0.5 * self.lateralAvg(D.v() * \
            christfl.v2d() * v.v() )

        Dh0.v()[:] += dt * 0.5 * self.lateralAvg(-(Dh.v() + \
            Dh0.v2d()) * christfl.v2d() * \
            v.v() )

        D.v()[:,:] -= dt * 0.5 * D.v() * v.v() * \
            christfl.v2d()

        Dh.v()[:,:] += dt * 0.5 * ( -(Dh.v() + Dh0.v2d()) * \
            v.v() * christfl.v2d())


        #fill ghostcells
        self.cc_data.fill_BC("enthalpy")
        self.cc_data.fill_BC("density")
        self.base_data.fill_BC("D0")
        self.base_data.fill_BC("Dh0")



    def EnforceHSE(self):
        """
        Enforce relativistic hydrostatic equilibrium on base pressure.
        """

        u0 = self.metric.calcu0()
        myg = self.cc_data.grid
        Dh0 = self.base_data.get_var("Dh0")
        g = self.rp.get_param("lm-atmosphere.grav")
        c = self.rp.get_param("lm-atmosphere.c")
        R = self.rp.get_param("lm-atmosphere.radius")

        # flatten u0 to make 1d for next equation.
        u0flat = self.lateralAvg(u0.v())
        p0 = self.base_data.get_var("p0")
        self.base_data.fill_BC("p0")
        for i in range(myg.jlo,myg.jhi+1):
            p0.d[i] = p0.d[i-1] - myg.dy * g * Dh0.d[i]/(u0flat[i] * \
                c**2 * self.metric.alpha.d[i]**2 * R)



    def preevolve(self):
        """
        preevolve is called before we being the timestepping loop.  For
        the low Mach solver, this does an initial projection on the
        velocity field and then goes through the full evolution to get the
        value of phi.  The fluid state (D, u, v) is then reset to values
        before this evolve.
        """

        myg = self.cc_data.grid

        Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")
        self.cc_data.fill_BC("enthalpy")

        coeff = self.aux_data.get_var("coeff")


        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # the coefficent for the elliptic equation is zeta^2/Dhu0
        # haven't evolved anything yet don't need to update zeta
        u0 = self.metric.calcu0()


        #Dh = np.ones(np.shape(Dh))
        #need some crazy buffer thing
        coeff.d[:,:] = 1.0 / (Dh.d * u0.d)
        zeta = self.base_data.get_var("zeta")

        try:
            coeff.v()[:,:] *= zeta.v2d()**2
        except FloatingPointError:
            print('zeta: ', np.max(zeta.d[:]))

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
        div_zeta_U.v()[:,:] = 0.5 * zeta.v2d() * (u.ip(1) - u.ip(-1))/myg.dx + \
                0.5 * (zeta.v2dp(1) * v.jp(1) - zeta.v2dp(-1) * v.jp(-1))/myg.dy

        # solve
        # D(zeta^2/Dhu0) G (phi/zeta) + zeta(S - dp/dt / Gamma1*p)= D( zeta U )

        # set the RHS to div_xi_U - xi(S -...) and solve
        constraint = self.calcConstraint(zeta)
        mg.init_RHS(div_zeta_U.v() -\
            constraint.v())
        mg.solve(rtol=1.e-12)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.cc_data.get_var("phi")
        phi.d[:,:] = mg.get_solution(grid=myg).d

        # get the cell-centered gradient of phi and update the
        # velocities
        # CHANGED: this update only needs to be done on the interior
        # cells -- not ghost cells
        gradp_x, gradp_y = mg.get_solution_gradient(grid=myg)

        coeff = 1.0/(Dh.d * u0.d)
        coeff.v()[:,:] *= zeta.v2d()

        u.v()[:,:] -= coeff.v() * gradp_x.v()
        v.v()[:,:] -= coeff.v() * gradp_y.v()

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

        orig_gp_x.d[:,:] = new_gp_x.d[:,:]
        orig_gp_y.d[:,:] = new_gp_y.d[:,:]

        self.cc_data = orig_data
        self.base_data = orig_base

        if self.verbose > 0: print("done with the pre-evolution")



    def evolve(self, dt):
        """
        Evolve the low Mach system through one timestep.

        Parameters
        ----------
        dt : float
            timestep
        """

        print('dt is: ', dt)

        D = self.cc_data.get_var("density")
        Dh = self.cc_data.get_var("enthalpy")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        gradp_x = self.cc_data.get_var("gradp_x")
        gradp_y = self.cc_data.get_var("gradp_y")

        # note: the base state quantities do not have valid ghost cells
        # need to update xi and zeta as D0, u0 etc change every time step
        self.updateXi(self.cc_data.grid)
        xi = self.base_data.get_var("xi")
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
        #FIXME: no longer use fortran for this, so can fix?
        D02d = np.array([D0.d,] * np.size(D0.d))
        Dh02d = np.array([Dh0.d,] * np.size(Dh0.d))

        # x slopes of r0, e0 surely just 0?
        ldelta_rx = limitFunc(1, D.d, myg)
        ldelta_ex = limitFunc(1, Dh.d, myg)
        ldelta_r0x = limitFunc(1, D02d.d, myg)
        ldelta_e0x = limitFunc(1, Dh02d.d, myg)
        ldelta_ux = limitFunc(1, u.d, myg)
        ldelta_vx = limitFunc(1, v.d, myg)

        ldelta_ry = limitFunc(2, D.d, myg)
        ldelta_ey = limitFunc(2, Dh.d, myg)
        ldelta_r0y = limitFunc(2, D02d.d, myg)
        ldelta_e0y = limitFunc(2, Dh02d.d, myg)
        ldelta_uy = limitFunc(2, u.d, myg)
        ldelta_vy = limitFunc(2, v.d, myg)


        g = self.rp.get_param("lm-atmosphere.grav")
        R = self.rp.get_param("lm-atmosphere.radius")
        c = self.rp.get_param("lm-atmosphere.c")

        #---------------------------------------------------------------------
        # React full and base states through first half timestep
        #---------------------------------------------------------------------

        self.ReactState(dt)

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


        # create the coefficient to the grad (pi/xi) term
        u0 = self.metric.calcu0()
        coeff = self.aux_data.get_var("coeff")
        tracer = self.cc_data.get_var("tracer")

        coeff.d[:,:] = 1.0/(Dh.d * u0.d)
        coeff.v()[:,:] *= xi.v2d()
        self.aux_data.fill_BC("coeff")


        # create the source term
        source = self.aux_data.get_var("source_y")

        """
        Have attempted to do this relativistically.

        Source here is given by -U_j Dlnu0/Dt + Gamma_{rho nu j} U^nu U^rho.
        The second term in our simple metric is just -g/c^2 R.
        The first term presents some difficulty so shall try to ignore it for
        now.
        """
        source.d[:,:] = -g  / (c**2 * R)
        self.aux_data.fill_BC("source_y")

        #u_MAC, v_MAC = lm_int.mac_vels(myg, dt, u, v, ldelta_ux, ldelta_vx,
        #                                       ldelta_uy, ldelta_vy,
        #                                       coeff*gradp_x, coeff*gradp_y,
        #                                       source)
        u_MAC, v_MAC = lm_interface_f.mac_vels(myg.qx, myg.qy, myg.ng,
                                               myg.dx, myg.dy, dt,
                                               u, v,
                                               ldelta_ux, ldelta_vx,
                                               ldelta_uy, ldelta_vy,
                                               gradp_x, gradp_y, coeff,
                                               source)

        #---------------------------------------------------------------------
        # do a MAC projection to make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve D (zeta^2/D) G phi = D (zeta U^MAC), where
        # phi is cell centered, and U^MAC is the MAC-type staggered
        # grid of the advective velocities.

        # create the coefficient array: zeta**2/Dhu0
        u0 = self.metric.calcu0()
        coeff.v(buf=1)[:,:] = 1.0/(Dh.v(buf=1) * \
            u0.v(buf=1))
        self.updateZeta(self.cc_data.grid)
        zeta = self.base_data.get_var("zeta")
        zeta_edges = self.base_data.get_var("zeta-edges")
        coeff.v(buf=1)[:,:] *= zeta.v2d(buf=1)**2

        if self.verbose > 0: print("  MAC projection")


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
                                 verbose=0, nsmooth=20)

        # first compute div{zeta U}
        div_zeta_U = mg.soln_grid.scratch_array()

        # MAC velocities are edge-centered.  div{zeta U} is cell-centred.
        div_zeta_U.v()[:,:] = zeta.v2d() * (
                u_MAC.ip(1) - u_MAC.v()) / myg.dx + \
            (zeta_edges.jp(1) * v_MAC.jp(1) -
             zeta_edges.v2d() * v_MAC.v()) / myg.dy

        # solve the Poisson problem
        constraint = self.calcConstraint(zeta.d)

        mg.init_RHS(div_zeta_U.d[:,:] - constraint.d)
        mg.solve(rtol=1.e-12)


        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/zeta
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC.d[:,:] = mg.get_solution(grid=myg).d

        coeff = self.aux_data.get_var("coeff")
        u0 = self.metric.calcu0()
        coeff.v()[:,:] = 1.0/(Dh.v() * u0.v())
        coeff.v()[:,:] *= zeta.v2d()
        self.aux_data.fill_BC("coeff")

        coeff_x = myg.scratch_array()
        b = (3, 1, 0, 0)  # this seems more than we need
        #coeff_x is edge-based, coeff is cell-centred.
        coeff_x.v(buf=b)[:,:] = 0.5 * (coeff.ip(-1, buf=b) + coeff.v(buf=b))

        #coeff_y is edge-based, coeff is cell-centred.
        coeff_y = myg.scratch_array()
        b = (0, 0, 3, 1)
        coeff_y.v(buf=b)[:,:] = 0.5 * (coeff.jp(-1, buf=b) + coeff.v(buf=b))

        # we need the MAC velocities on all edges of the computational domain
        # here we do U = U - (zeta/D) grad (phi/zeta)
        b = (0, 1, 0, 0)
        u_MAC.v(buf=b)[:,:] -= coeff_x.v(buf=b) * (phi_MAC.v(buf=b) - \
                phi_MAC.ip(-1, buf=b)) / myg.dx

        b = (0, 0, 0, 1)
        v_MAC.v(buf=b)[:,:] -= coeff_y.v(buf=b) * (phi_MAC.v(buf=b) - \
                phi_MAC.jp(-1, buf=b)) / myg.dy


        # resize so that Fortran is happy
        D02d.d[:,:] = np.array([D0.d,] * np.size(D0.d))
        Dh02d.d[:,:] = np.array([Dh0.d,] * np.size(Dh0.d))


        #---------------------------------------------------------------------
        # predict D to the edges and do its conservative update
        #---------------------------------------------------------------------
        D_xint, D_yint = lm_int.D_states(myg, dt, D,
                            u_MAC, v_MAC, ldelta_rx, ldelta_ry)

        _, D0_yint = lm_int.D_states(myg, dt, D02d, u_MAC,
                            v_MAC, ldelta_r0x, ldelta_r0y)

        #D_xint, D_yint are on edges, D cell-centred.
        D.v()[:,:] -= dt*(
            #  (D u)_x
            (D_xint.ip(1) * u_MAC.ip(1) - D_xint.v() * u_MAC.v())/myg.dx +
            #  (D v)_y
            (D_yint.jp(1) * v_MAC.jp(1) - D_yint.v() * v_MAC.v())/myg.dy )

        #self.checkXSymmetry(D, myg.qx)
        #print('D symmetry')


        #need to do some averaging as D0 is only 1d

        D02d.v()[:] -= dt*(
            #  (D0 u)_x
            #(D0_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]*
            # u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             #D0_xint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             #u_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx +
            #  (D0 v)_y
            (D0_yint.jp(1) * v_MAC.jp(1) - D0_yint.v() * v_MAC.v()) / myg.dy )

        D0.d[:] = self.lateralAvg(D02d.d)


        #---------------------------------------------------------------------
        # predict Dh to the edges and do its conservative update
        #
        # Exactly the same as for density
        #---------------------------------------------------------------------
        Dh_xint, Dh_yint = lm_int.D_states(myg, dt, D, u_MAC, v_MAC, ldelta_ex,
                            ldelta_ey)

        _, Dh0_yint = lm_int.D_states(myg, dt, Dh02d, u_MAC,
                            v_MAC, ldelta_e0x, ldelta_e0y)

        Dh_old = Dh.copy()

        Dh.v()[:,:] -= dt * (
            #  (Dh u)_x
            (Dh_xint.ip(1) * u_MAC.ip(1) - Dh_xint.v() * u_MAC.v()) / myg.dx +
            #  (Dh v)_y
            (Dh_yint.jp(1) * v_MAC.jp(1) - Dh_yint.v() * v_MAC.v()) / myg.dy )

        # average laterally as Dh0 is 1d

        Dh02d.v()[:,:] -= dt * (
            #  (Dh u)_x
            #(Dh0_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]*
             #u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             #Dh0_xint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             #u_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx +
            #  (Dh v)_y
            (Dh0_yint.jp(1) * v_MAC.jp(1) - Dh0_yint.v() * v_MAC.v()) / myg.dy )

        Dh0.d[:] = self.lateralAvg(Dh02d.d)

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")
        self.base_data.fill_BC("D0")
        self.base_data.fill_BC("Dh0")

        #---------------------------------------------------------------------
        # Enforce relativistic hydrostatic equilibrium on base pressure
        #---------------------------------------------------------------------

        self.EnforceHSE()


        #---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------

        if self.verbose > 0: print("  making u, v edge states")


        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:,:] = 2./((Dh.v() + Dh_old.v()) * u0.v())

        # Might not need to recalculate zeta but shall just in case
        self.updateZeta(self.cc_data.grid)
        zeta = self.base_data.get_var("zeta")
        coeff.d[:,:] *= zeta.v2d(buf=zeta.ng)
        self.aux_data.fill_BC("coeff")

        u_xint, v_xint, u_yint, v_yint = \
            lm_int.states(myg, dt, u, v, ldelta_ux, ldelta_vx,
                                 ldelta_uy, ldelta_vy,
                                 gradp_x, gradp_y,
                                 coeff, source, u_MAC, v_MAC)


        #---------------------------------------------------------------------
        # update U to get the provisional velocity field
        #---------------------------------------------------------------------

        if self.verbose > 0: print("  doing provisional update of u, v")


        # compute (U.grad)U

        # we want u_MAC U_x + v_MAC U_y
        advect_x = myg.scratch_array()
        advect_y = myg.scratch_array()

        advect_x.v()[:,:] = 0.5 * (u_MAC.v() + u_MAC.ip(1)) * \
            (u_xint.ip(1) - u_xint.v()) / myg.dx + \
            0.5 * (v_MAC.v() +  v_MAC.jp(1)) * \
            (u_yint.jp(1) - u_yint.v())/myg.dy

        advect_y.v()[:,:] = 0.5 * (u_MAC.v() + u_MAC.ip(1)) * \
            (v_xint.ip(1) - v_xint.v())  /myg.dx + \
            0.5 * (v_MAC.v() + v_MAC.jp(1)) * \
            (v_yint.jp(1) - v_yint.v())/myg.dy


        proj_type = self.rp.get_param("lm-atmosphere.proj_type")


        if proj_type == 1:
            u.v()[:,:] -= (dt * advect_x.v() + dt * gradp_x.v())

            v.v()[:,:] -= (dt * advect_y.v() + dt * gradp_y.v())

        elif proj_type == 2:
            u.v()[:,:] -= dt * advect_x.v()

            v.v()[:,:] -= dt * advect_y.v()

        # add the gravitational source (and pressure source)
        u0 = self.metric.calcu0()
        gamma = self.rp.get_param("eos.gamma")
        #p0 = self.base_data.get_var("p0")
        self.updateXi(self.cc_data.grid)
        xi = self.base_data.get_var("xi")
        gradpi_x = myg.scratch_array()
        gradpibyxi_y = myg.scratch_array()
        xi2d = np.array([xi.d,] * np.size(xi.d))

        #predict xi to edges by first finding limited slopes
        ldelta_xix = limitFunc(1, xi2d, myg)
        ldelta_xiy = limitFunc(2, xi2d, myg)

        # FIXME: should you use MAC velocities/phi or advected ones??
        _, xi_yint = lm_int.D_states(myg, dt, xi2d, u_MAC, v_MAC,
                                           ldelta_xix, ldelta_xiy)

        #edge-centered pi given by phi_MAC/delta t, so find gradpi:
        gradpi_x.v()[:,:] = (phi_MAC.ip(1) - phi_MAC.v()) / (myg.dx * dt)
        gradpibyxi_y.v()[:,:] = (phi_MAC.jp(1) / xi_yint.jp(1) - \
            phi_MAC.v() / xi_yint.v()) / (myg.dy * dt)


        pressureSource = myg.scratch_array()
        drp0 = myg.scratch_array()

        # drp0 + xi*dr(pi/xi)
        # cell-centred
        #drp0[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
        #    0.5 * (p0[np.newaxis, myg.jlo+1:myg.jhi+2] - \
        #    p0[np.newaxis, myg.jlo-1:myg.jhi]) / myg.dy

        drp0.v()[:,:] = -Dh0.v2d() * g / \
            (u0.v() * c**2 * R * self.metric.alpha.v2d()**2)

        # cell-centred
        pressureSource.v()[:,:] = drp0.v() + xi.v2d() * gradpibyxi_y.v()

        print("min/max u   = {}, {}".format(np.min(u.d), np.max(u.d)))
        print("min/max v   = {}, {}".format(np.min(v.d), np.max(v.d)))
        print("min/max u0   = {}, {}".format(np.min(u0.d), np.max(u0.d)))

        # dx pi
        #cell-centred
        u.v()[:,:] += dt * -gradpi_x.v() / (Dh.v() * u0.v())

        v.v()[:,:] += dt * (-pressureSource.v() / (Dh.v() * u0.v()) - \
            g / (c**2 * R))

        print('calculated u and v')

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        print("min/max D = {}, {}".format(np.min(D.d), np.max(D.d)))
        print("min/max Dh = {}, {}".format(np.min(Dh.d), np.max(Dh.d)))
        print("min/max u   = {}, {}".format(np.min(u.d), np.max(u.d)))
        print("min/max v   = {}, {}".format(np.min(v.d), np.max(v.d)))


        # calculate the sound speed using base density as
        gamma = self.rp.get_param("eos.gamma")
        u0 = self.metric.calcu0()
        cs = gamma * (D0.v2d(buf=D0.ng)/u0.d)**(gamma - 1.)
        cs[:,:] = np.sqrt(np.abs(cs[:,:]))

        print("min/max c_s   = {}, {}".format(np.min(cs), np.max(cs)))

        # calculate and print Mach number
        speed = np.sqrt(u.d**2 + v.d**2) * u0.d
        M = speed[:,:]/cs[np.newaxis,:]
        print("min/max M   = {}, {}".format(np.min(M), np.max(M)))

        if np.max(M) > 0.2:
            msg.bold('\nWarning!')
            print('Mach number is ', np.max(M), '- simulation will be less accurate.\n')

        #---------------------------------------------------------------------
        # React full and base states through second half timestep
        #---------------------------------------------------------------------

        self.ReactState(dt)

        #---------------------------------------------------------------------
        # project the final velocity
        #---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)

        # create the coefficient array: zeta**2/Dhu0
        u0 = self.metric.calcu0()
        coeff.d = 1.0/(Dh.d * u0.d)
        self.updateZeta(self.cc_data.grid)
        zeta = self.base_data.get_var("zeta")
        coeff.v()[:,:] *= zeta.v2d()**2

        if self.verbose > 0: print("  final projection")

        # first compute div{zeta U}

        # u/v are cell-centered, divU is cell-centered
        div_zeta_U.v()[:,:] = 0.5 * zeta.v2d() * \
                (u.ip(1) - u.ip(-1)) / myg.dx + \
            0.5 * (zeta.v2dp(1) * v.jp(1) - zeta.v2dp(-1) * v.jp(-1)) / myg.dy

        constraint = self.calcConstraint(zeta)
        mg.init_RHS((div_zeta_U.d- constraint.d) / dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess.v(buf=1)[:,:] = phi.v(buf=1)
        mg.init_solution(phiGuess)

        # solve
        mg.solve(rtol=1.e-12)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi.d[:,:] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)

        # U = U - (zeta/Dhu0) grad (phi/zeta)
        u0 = self.metric.calcu0()
        coeff.d[:,:] = 1.0/(Dh.d * u0.d)
        coeff.v()[:,:] *= zeta.v2d()

        u.v()[:,:] -= dt * coeff.v() * gradphi_x.v()

        v.v()[:,:] -= dt * coeff.v() * gradphi_y.v()

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

        tracer.d[:,:] = v.copy()




    def dovis(self):
        """
        Do runtime visualization
        """
        plt.clf()

        #plt.rc("font", size=10)

        D = self.cc_data.get_var("density")
       # Dh = self.cc_data.get_var("enthalpy")
        #D0 = self.base_data.get_var("D0")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        #self.metric.calcu0()
        #tracer = self.cc_data.get_var("tracer")
        #print('xi size', np.shape(xi))

        myg = self.cc_data.grid
        u0 = np.ones((myg.qx, myg.qy))

        magvel = np.sqrt(u.d**2 + v.d**2) / u0.d

        #gamma = self.rp.get_param("eos.gamma")
        gamma = 2.6
        #cs[:,:] = gamma * (gamma - 1.) / (2. - gamma)
        #cs[:,:] *= (D[:,:] + Dh[:,:]) / D[:,:]
        #cs[:,:] = np.sqrt(np.abs(cs[:,:]))
        cs = gamma * (D.d / u0.d)**(gamma - 1.)
        cs[:] = np.sqrt(np.abs(cs[:]))
       # M = magvel[:,:] / cs[np.newaxis,:]

        vort = myg.scratch_array()

        dv = 0.5 * (v.ip(1) - v.ip(-1)) / myg.dx

        du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy


        # for some reason, setting vort here causes the density in the
        # simulation to NaN.  Seems like a bug (in python?)
        vort.v()[:,:] = (dv - du) * u0.v()

        _, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.25)

        fields = [D, magvel, u, v]
        field_names = [r"$D$", r"$|U|$", r"$U$", r"$V$"]

        for n in range(len(fields)):

            ax = axes.flat[n]

            f = fields[n]

            if n < 2:
                cmap = plt.cm.jet
                vmin = np.min(f.v())
                vmax = np.max(f.v())
            else:
                cmap = plt.cm.seismic
                vmin = -np.max(np.abs(f.v()))
                vmax = +np.max(np.abs(f.v()))

            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap = cmap, vmin = vmin, vmax = vmax)

            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title(field_names[n])

            plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125, "n: %4d,   t = %10.5f" % (self.cc_data.n, self.cc_data.t))
        #plt.tight_layout()

        plt.draw()



    #def finalize(self):
    #    """
    #    Do any final clean-ups for the simulation and call the problem's
    #    finalize() method.
    #    """
    #    exec(self.problem_name + '.finalize()')
