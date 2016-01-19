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
from lm_gr.simulation import *
from lm_gr.simulation_react import *
import multigrid.variable_coeff_MG as vcMG
import multigrid.rect_MG as rectMG
from util import profile
import lm_gr.metric as metric
import colormaps as cmaps


class SimulationSpherical(SimulationReact):

    def __init__(self, solver_name, problem_name, rp, timers=None, fortran=True):
        """
        Initialize the SimulationSpherical object

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
            Determines whether to use the fortran smoother or the original
            python one.
        """

        SimulationReact.__init__(self, solver_name, problem_name, rp, timers=timers, fortran=fortran)

        self.r2d = []
        self.r2v = []


    def initialize(self):
        """
        Initialize the grid and variables for low Mach general relativistic atmospheric flow
        and set the initial conditions for the chosen problem.
        """

        super(SimulationSpherical, self).initialize()

        R = self.rp.get_param("lm-gr.radius")
        myg = self.cc_data.grid

        self.r2d = myg.y2d + R
        self.r2v = r2d[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1]

        # set up spherical metric
        alpha = Basestate(myg.ny, ng=myg.ng)

        # r = y + R, where r is measured from the centre of the star,
        # R is the star's radius and y is measured from the surface
        alpha.d[:] = np.sqrt(1. - 2. * g * (1. - myg.y[:]/R) / (R * c**2))

        beta = [0., 0.]

        gamma_matrix = np.zeros((myg.qx, myg.qy, 2, 2), dtype=np.float64)
        gamma_matrix[:,:,:,:] = 1. + 2. * g * \
            (1. - myg.y[np.newaxis, :, np.newaxis, np.newaxis] / R) / \
            (R * c**2) * np.eye(2)[np.newaxis, np.newaxis, :, :]

        gamma_matrix[:,:,1,1] *= (myg.y2d + R)

        self.metric = metric.Metric(self.cc_data, self.rp, alpha, beta,
                                    gamma_matrix, cartesian=False)

        u0 = self.metric.calcu0()


    def calc_psi(self, S=None, U0=None, p0=None, old_p0=None):
        r"""
        Calculate :math: `\psi`
        .. math::

            \psi = \partial_tp_0 + U_0\partial_r p_0

        Parameters
        ----------
        S : ArrayIndexer object, optional
            source term
        U0 : ArrayIndexer object, optional
            base state of the radial velocity
        p0 : ArrayIndexer object, optional
            base state pressure
        p0_old: ArrayIndexer object, optional
            previous base state pressure

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

        psi = super(SimulationSpherical, self).calc_psi(S=S, U0=U0, p0=p0, old_p0=old_p0)

        #psi.v(buf=myg.ng-1)[:] = gamma * 0.25 * \
        #    (p0.v(buf=myg.ng-1) + old_p0.v(buf=myg.ng-1)) * \
        #    (self.lateral_average(S.v(buf=myg.ng-1)) -
        #     (U0.jp(1, buf=myg.ng-1) - U0.jp(-1, buf=myg.ng-1)))

        # really confused what this does?

        return psi


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
        mg = rectMG.RectMG2d(myg.nx, myg.ny,
                             xl_BC_type=self.aux_data.BCs["phi"].xlb,
                             xr_BC_type=self.aux_data.BCs["phi"].xrb,
                             yl_BC_type=self.aux_data.BCs["phi"].ylb,
                             yr_BC_type=self.aux_data.BCs["phi"].yrb,
                             xmin=myg.xmin, xmax=myg.xmax,
                             ymin=myg.ymin, ymax=myg.ymax,
                             coeffs=coeff,
                             coeffs_bc=self.cc_data.BCs["density"],
                             verbose=0)

        # first compute div{zeta U}
        div_zeta_U = mg.soln_grid.scratch_array()

        # u/v are cell-centered, divU is cell-centered
        div_zeta_U.v()[:,:] = \
            0.5 * zeta.v2df(myg.qx) * (u.ip(1) - u.ip(-1)) / (myg.dx * self.r2v) + \
            0.5 * (zeta.v2dpf(myg.qx, 1) * v.jp(1) - \
            zeta.v2dpf(myg.qx, -1) * v.jp(-1)) / myg.dy +  zeta.v2df(myg.qx) * u.v() / (self.r2v * np.tan(myg.x2v)) + 2. * zeta.v2df(myg.qx) * v.v() / self.r2v

        # solve D (zeta^2/Dh u0) G (phi/zeta) = D( zeta U )
        constraint = self.constraint_source()
        # set the RHS to divU and solve
        mg.init_RHS(div_zeta_U.v(buf=1) - constraint.v(buf=1))
        mg.solve(rtol=1.e-12, fortran=self.fortran)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.aux_data.get_var("phi")
        phi.d[:,:] = mg.get_solution(grid=myg).d

        # get the cell-centered gradient of phi and update the
        # velocities
        # FIXME: this update only needs to be done on the interior
        # cells -- not ghost cells
        gradp_x, gradp_y = mg.get_solution_gradient(grid=myg)
        #pdb.set_trace()

        coeff = 1. / (Dh * u0)
        coeff.v()[:,:] *= zeta.v2d()

        ###########################

        ##### GRADP USED HERE #####

        ###########################
        #gradp_x.d[:,:] = 1.
        #gradp_y.d[:,:] = 1.

        # CHANGED: multiplied by dt to match same thing done at end of evolve.
        # FIXME: WTF IS IT DOING HERE?????
        u.v()[:,:] -= self.dt * coeff.v() * gradp_x.v()
        #v.v()[:,:] -= self.dt * coeff.v() * gradp_y.v()

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        # c. now get an approximation to gradp at n-1/2 by going through the
        # evolution.

        # store the current solution -- we'll restore it in a bit
        orig_data = patch.cell_center_data_clone(self.cc_data)
        orig_aux = patch.cell_center_data_clone(self.aux_data)

        # get the timestep
        self.compute_timestep(u0=u0)

        # evolve
        self.evolve()

        # update gradp_x and gradp_y in our main data object
        new_gp_x = self.aux_data.get_var("gradp_x")
        new_gp_y = self.aux_data.get_var("gradp_y")

        orig_gp_x = orig_aux.get_var("gradp_x")
        orig_gp_y = orig_aux.get_var("gradp_y")

        orig_gp_x.d[:,:] = new_gp_x.d[:,:]
        orig_gp_y.d[:,:] = new_gp_y.d[:,:]

        self.cc_data = orig_data
        self.aux_data = orig_aux

        v = self.cc_data.get_var("y-velocity")

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

        gradp_x = self.aux_data.get_var("gradp_x")
        gradp_y = self.aux_data.get_var("gradp_y")
        DX = self.cc_data.get_var("mass-frac")
        scalar = self.cc_data.get_var("scalar")
        T = self.cc_data.get_var("temperature")

        u0 = self.metric.calcu0()

        # note: the base state quantities do not have valid ghost cells
        self.update_zeta(u0=u0)
        zeta = self.base["zeta"]
        zeta_edges = self.base["zeta-edges"]

        D0 = self.base["D0"]
        Dh0 = self.base["Dh0"]
        U0 = self.base["U0"]

        phi = self.aux_data.get_var("phi")

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

        ldelta_r0x = limitFunc(1, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_e0x = limitFunc(1, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_ux = limitFunc(1, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v.d, myg.qx, myg.qy, myg.ng)

        ldelta_r0y = limitFunc(2, D0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_e0y = limitFunc(2, Dh0.d2df(myg.qx), myg.qx, myg.qy, myg.ng)
        ldelta_uy = limitFunc(2, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v.d, myg.qx, myg.qy, myg.ng)

        #---------------------------------------------------------------------
        # 1. React state through dt/2
        #---------------------------------------------------------------------
        D_1 = myg.scratch_array(data=D.d)
        Dh_1 = myg.scratch_array(data=Dh.d)
        DX_1 = myg.scratch_array(data=DX.d)
        scalar_1 = myg.scratch_array(data=scalar.d)
        T_1 = myg.scratch_array(data=T.d)
        self.react_state(D=D_1, Dh=Dh_1, DX=DX_1, T=T_1, scalar=scalar_1, u0=u0)

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

        ###########################

        ##### GRADP USED HERE #####

        ###########################
        #gradp_x.d[:,:] = 1.
        #gradp_y.d[:,:] = 1.
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
        mg = rectMG.RectMG2d(myg.nx, myg.ny,
                             xl_BC_type=self.aux_data.BCs["phi-MAC"].xlb,
                             xr_BC_type=self.aux_data.BCs["phi-MAC"].xrb,
                             yl_BC_type=self.aux_data.BCs["phi-MAC"].ylb,
                             yr_BC_type=self.aux_data.BCs["phi-MAC"].yrb,
                             xmin=myg.xmin, xmax=myg.xmax,
                             ymin=myg.ymin, ymax=myg.ymax,
                             coeffs=coeff,
                             coeffs_bc=self.cc_data.BCs["density"],
                             verbose=0)

        # first compute div{zeta U}
        div_zeta_U = mg.soln_grid.scratch_array()

        # MAC velocities are edge-centered.  div{zeta U} is cell-centered.
        div_zeta_U.v()[:,:] = \
            zeta.v2d() * (u_MAC.ip(1) - u_MAC.v()) / (myg.dx + \
            self.r2v) + zeta.v2d() * u.v() / (np.tan(myg.x2v) * \
            self.r2v) + \
            (zeta_edges.v2dp(1) * v_MAC.jp(1) - \
            zeta_edges.v2d() * v_MAC.v()) / myg.dy + \
            2. * zeta.v2d() * v.v() / self.r2v

        # careful: this makes u0_MAC edge-centred.
        u0_MAC = self.metric.calcu0(u=u_MAC, v=v_MAC)
        constraint = self.constraint_source(u=u_MAC, v=v_MAC, S=S_t_centred)

        # solve the Poisson problem
        mg.init_RHS(div_zeta_U.d - constraint.v(buf=1))
        mg.solve(rtol=1.e-12, fortran=self.fortran)

        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/zeta
        #phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC = self.aux_data.get_var("phi-MAC")
        phi_MAC.d[:,:] = mg.get_solution(grid=myg).d
        gradp_MAC_x, gradp_MAC_y = mg.get_solution_gradient(grid=myg)

        coeff = self.aux_data.get_var("coeff")
        # FIXME: is this u0 or u0_MAC?
        coeff.d[:,:] = self.metric.alpha.d2d()**2 / (Dh.d * u0.d)
        coeff.d[:,:] *= zeta.d2d()
        self.aux_data.fill_BC("coeff")

        coeff_x = myg.scratch_array()
        b = (3, 1, 0, 0)  # this seems more than we need
        coeff_x.v(buf=b)[:,:] = 0.5 * (coeff.ip(-1, buf=b) + coeff.v(buf=b))
        gradp_MAC_x.v(buf=b)[:,:] = 0.5 * (gradp_MAC_x.ip(-1, buf=b) + gradp_MAC_x.v(buf=b))

        coeff_y = myg.scratch_array()
        b = (0, 0, 3, 1)
        coeff_y.v(buf=b)[:,:] = 0.5 * (coeff.jp(-1, buf=b) + coeff.v(buf=b))
        gradp_MAC_y.v(buf=b)[:,:] = 0.5 * (gradp_MAC_y.jp(-1, buf=b) + gradp_MAC_y.v(buf=b))

        #############################

        ##### phi_MAC USED HERE #####

        #############################
        #phi_MAC.d[:,:] = 1.

        # we need the MAC velocities on all edges of the computational domain
        # here we do U = U - (zeta/Dh u0) grad (phi/zeta)
        b = (0, 1, 0, 0)
        #u_MAC.v(buf=b)[:,:] -= coeff_x.v(buf=b) * \
        #    (phi_MAC.v(buf=b) - phi_MAC.ip(-1, buf=b)) / myg.dx
        u_MAC.v(buf=b)[:,:] -= coeff_x.v(buf=b) * \
            gradp_MAC_x.v(buf=b)

        b = (0, 0, 0, 1)
        #v_MAC.v(buf=b)[:,:] -= coeff_y.v(buf=b) * \
        #    (phi_MAC.v(buf=b) - phi_MAC.jp(-1, buf=b)) / myg.dy
        v_MAC.v(buf=b)[:,:] -= coeff_y.v(buf=b) * \
            gradp_MAC_y.v(buf=b)

        #u0_MAC = self.metric.calcu0(u=u_MAC, v=v_MAC)
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
        ldelta_px = limitFunc(1, psi_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_py = limitFunc(2, psi_1.d, myg.qx, myg.qy, myg.ng)
        no_source = myg.scratch_array()
        _px, _py = lm_interface_f.psi_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             psi_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_px, ldelta_py, no_source.d)


        X_1 = myg.scratch_array(data=DX_1.d/D_1.d)
        ldelta_Xx = limitFunc(1, X_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_Xy = limitFunc(2, X_1.d, myg.qx, myg.qy, myg.ng)
        _, omega_dot = self.calc_Q_omega_dot(D=D_1, DX=DX_1, u=u_MAC, v=v_MAC, u0=u0_MAC, T=T_1)
        _Xx, _Xy = lm_interface_f.psi_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             X_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_Xx, ldelta_Xy, omega_dot.d)
        # x component of U0 is zero
        U0_x = myg.scratch_array()
        # is U0 edge-centred?
        _r0x, _r0y = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng,
                                            myg.dx, myg.dy, self.dt,
                                            D0.d2df(myg.qx), U0_x.d, U0_half_star.d2df(myg.qx),
                                            ldelta_r0x, ldelta_r0y)

        D0_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()[:,:]
        D02d.v()[:,:] -= self.dt*(
            #  (D v)_y
            (D0_yint.jp(1) * U0_half_star.jp(1)[np.newaxis,:] - \
             D0_yint.v() * U0_half_star.v2d())/myg.dy + 2. * D0.v2d() * U0_half_star.v2d() / self.r2v)

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

        X_xint = patch.ArrayIndexer(d=_Xx, grid=myg)
        X_yint = patch.ArrayIndexer(d=_Xy, grid=myg)

        D_xint.d[:,:] += 0.5 * (D0_xint.d + D0_2a_xint.d)
        D_yint.d[:,:] += 0.5 * (D0_yint.d + D0_2a_yint.d)

        scalar_xint = myg.scratch_array(data=psi_xint.d*D_xint.d)
        scalar_yint = myg.scratch_array(data=psi_yint.d*D_yint.d)

        DX_xint = myg.scratch_array(data=X_xint.d*D_xint.d)
        DX_yint = myg.scratch_array(data=X_yint.d*D_yint.d)

        D_old = myg.scratch_array(data=D.d)
        scalar_2_star = myg.scratch_array(data=scalar_1.d)
        D_2_star = myg.scratch_array(data=D_1.d)
        DX_2_star = myg.scratch_array(data=DX_1.d)

        scalar_2_star.v()[:,:] -= self.dt * (
            #  (psi D u)_x
            (scalar_xint.ip(1) * u_MAC.ip(1) - scalar_xint.v() * u_MAC.v())/(myg.dx * self.r2v) + scalar_1.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (psi D v)_y
            (scalar_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -
             scalar_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy  + 2. * scalar_1.v() * (v.v() + U0_half_star.v2d()) / self.r2v)

        DX_2_star.v()[:,:] -= self.dt * (
            #  (X D u)_x
            (DX_xint.ip(1) * u_MAC.ip(1) - DX_xint.v() * u_MAC.v())/(myg.dx * self.r2v) + DX_1.v() * u.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (X D v)_y
            (DX_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -
             DX_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy  + 2. * DX_1.v() * (v.v() + U0_half_star.v2d()) / self.r2v)

        D_2_star.v()[:,:] -= self.dt * (
            #  (D u)_x
            (D_xint.ip(1) * u_MAC.ip(1) - D_xint.v() * u_MAC.v())/(myg.dx * self.r2v) + D_1.v() * u.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (D v)_y
            (D_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -
             D_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy + 2. * D_1.v() * (v.v() + U0_half_star.v2d()) / self.r2v)

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

        Dh02d = myg.scratch_array()
        Dh02d.d[:,:] = Dh0.d2d()
        Dh02d.v()[:,:] += -self.dt * (
            #  (D v)_y
            (Dh0_yint.jp(1) * U0_half_star.jp(1)[np.newaxis,:] - \
             Dh0_yint.v() * U0_half_star.v2d())/myg.dy + 2. * Dh0.v2df(myg.qx) * (v.v() + U0_half_star.v2d()) / self.r2v) + \
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
        Dh_2_star = myg.scratch_array(data=Dh_1.d)
        # Dh0 is not edge based?
        drp0 = self.drp0(Dh0=Dh0, u=u_MAC, v=v_MAC, u0=u0_MAC)

        # 4Hii.
        Dh_2_star.v()[:,:] += -self.dt * (
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())/(myg.dx * self.r2v) + Dh_1.v() * u.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (D v)_y
            (Dh_yint.jp(1)*(v_MAC.jp(1) + U0_half_star.jp(1)[np.newaxis,:]) -\
             Dh_yint.v() * (v_MAC.v() + U0_half_star.v2d())) / myg.dy  + \
            + 2. * Dh_1.v() * (v.v() + U0_half_star.v2d()) / self.r2v) +\
            self.dt * u0_MAC.v() * v_MAC.v() * drp0.v2d() + \
            self.dt * u0_MAC.v() * psi.v()

        self.cc_data.fill_BC("enthalpy")

        # this makes p0 -> p0_star. May not want to update self.base[p0] here.
        p0 = self.base["p0"]
        p0_star = Basestate(myg.ny, ng=myg.ng)
        p0_star.d[:] = p0.d
        self.enforce_tov(p0=p0_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC, u0=u0_MAC)

        # update eint as a diagnostic
        eint = self.aux_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()

        # update T based on EoS
        T_2_star = myg.scratch_array()
        self.calc_T(p0=p0_star, D=D_2_star, DX=DX_2_star, u=u_MAC, v=v_MAC, u0=u0_MAC, T=T_2_star)


        #---------------------------------------------------------------------
        # 5. React state through dt/2
        #---------------------------------------------------------------------
        D_star = myg.scratch_array(data=D_2_star.d)
        Dh_star = myg.scratch_array(data=Dh_2_star.d)
        DX_star = myg.scratch_array(data=DX_2_star.d)
        scalar_star = myg.scratch_array(data=scalar_2_star.d)
        T_star = myg.scratch_array(data=T_2_star.d)
        self.react_state(S=self.compute_S(u=u_MAC, v=v_MAC),
                         D=D_star, Dh=Dh_star, DX=DX_star, T=T_star, scalar=scalar_star, Dh0=Dh0_star, u=u_MAC, v=v_MAC, u0=u0_MAC)

        #---------------------------------------------------------------------
        # 6. Compute time-centred expasion S, base state velocity U0 and
        # base state forcing
        #---------------------------------------------------------------------
        Q_2_star, _ = self.calc_Q_omega_dot(D=D_2_star, DX=DX_2_star, u=u_MAC, v=v_MAC, u0=u0_MAC, T=T_2_star)
        S_star = self.compute_S(u=u_MAC, v=v_MAC, u0=u0_MAC, Q=Q_2_star, D=D_2_star)

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

        # advect MAC velocities to get cell-centred quantities

        advect_x.v()[:,:] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(u_xint.ip(1) - u_xint.v())/myg.dx +\
            0.5*(v_MAC.v() + v_MAC.jp(1))*(u_yint.jp(1) - u_yint.v())/myg.dy

        advect_y.v()[:,:] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(v_xint.ip(1) - v_xint.v())/myg.dx +\
            0.5*(v_MAC.v() + v_MAC.jp(1))*(v_yint.jp(1) - v_yint.v())/myg.dy

        proj_type = self.rp.get_param("lm-gr.proj_type")

        ###########################

        ##### GRADP USED HERE #####

        ###########################
        #gradp_x.d[:,:] = 1.
        #gradp_y.d[:,:] = 1.
        # THIS TERM DOES NOT BREAK THINGS
        if proj_type == 1:
            u.v()[:,:] -= (self.dt * advect_x.v() + self.dt * gradp_x.v())
            # FIXME: add back in!
            #v.v()[:,:] -= (self.dt * advect_y.v() + self.dt * gradp_y.v())

        elif proj_type == 2:
            u.v()[:,:] -= self.dt * advect_x.v()
            # FIXME: add back in!
            #v.v()[:,:] -= self.dt * advect_y.v()

        # add on source term
        # do we want to use Dh half star here maybe?
        # FIXME: u_MAC, v_MAC in source?? No: MAC quantities are edge based, this is cell-centred
        mom_source_x, mom_source_r = self.calc_mom_source(Dh=Dh_star, Dh0=Dh0_star, u=u, v=v, u0=u0)
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
            print("min/max T = {}, {}".format(self.cc_data.min("temperature"), self.cc_data.max("temperature")))
            print("mean X   = {}".format(np.mean(DX.d/D.d)))

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
                                             ldelta_px, ldelta_py, no_source.d)

        X_1.d[:,:] = DX_1.d / D_1.d
        ldelta_Xx = limitFunc(1, X_1.d, myg.qx, myg.qy, myg.ng)
        ldelta_Xy = limitFunc(2, X_1.d, myg.qx, myg.qy, myg.ng)
        _, omega_dot = self.calc_Q_omega_dot(D=D_1, DX=DX_1, u=u_MAC, v=v_MAC, u0=u0_MAC, T=T_1)
        _Xx, _Xy = lm_interface_f.psi_states(myg.qx, myg.qy, myg.ng,
                                             myg.dx, myg.dy, self.dt,
                                             X_1.d, u_MAC.d, v_MAC.d,
                                             ldelta_Xx, ldelta_Xy, omega_dot.d)

        D0_xint = patch.ArrayIndexer(d=_r0x, grid=myg)
        D0_yint = patch.ArrayIndexer(d=_r0y, grid=myg)

        D02d = myg.scratch_array()
        D02d.d[:,:] = D0.d2d()
        D02d.v()[:,:] -= self.dt * (
            #  (D v)_y
            (D0_yint.jp(1) * U0_half.jp(1)[np.newaxis,:] -
             D0_yint.v() * U0_half.v2d())/myg.dy + 2. * D0.v2df(myg.qx) * (v.v() + U0_half.v2d()) / self.r2v)

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

        X_xint = patch.ArrayIndexer(d=_Xx, grid=myg)
        X_yint = patch.ArrayIndexer(d=_Xy, grid=myg)

        D_xint.d[:,:] += 0.5 * (D0_xint.d + D0_2a_xint.d)
        D_yint.d[:,:] += 0.5 * (D0_yint.d + D0_2a_yint.d)

        scalar_xint.d[:,:] = D_xint.d * psi_xint.d
        scalar_yint.d[:,:] = D_yint.d * psi_yint.d

        DX_xint.d[:,:] = X_xint.d * D_xint.d
        DX_yint.d[:,:] = X_yint.d * D_yint.d

        D_old = myg.scratch_array(data=D.d)
        scalar_2 = myg.scratch_array(data=scalar_1.d)
        D_2 = myg.scratch_array(data=D_1.d)
        DX_2 = myg.scratch_array(data=DX_1.d)

        D_2.v()[:,:] -= self.dt * (
            #  (D u)_x
            (D_xint.ip(1)*u_MAC.ip(1) - D_xint.v()*u_MAC.v())/(myg.dx * self.r2v) + D_1.v() * u.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (D v)_y
            (D_yint.jp(1)*(v_MAC.jp(1) + U0_half.jp(1)[np.newaxis,:]) -\
             D_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy + 2. * D_1.v() * (v.v() + U0_half.v2d()) / self.r2v)

        DX_2.v()[:,:] -= self.dt * (
            #  (X D u)_x
            (DX_xint.ip(1) * u_MAC.ip(1) - DX_xint.v() * u_MAC.v())/(myg.dx * self.r2v) + DX_1.v() * u.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (X D v)_y
            (DX_yint.jp(1)*(v_MAC.jp(1) + U0_half.jp(1)[np.newaxis,:]) -
             DX_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy + 2. * DX_1.v() * (v.v() + U0_half.v2d()) / self.r2v)

        scalar_2.v()[:,:] -= self.dt * (
            #  (D u)_x
            (scalar_xint.ip(1) * u_MAC.ip(1) -
             scalar_xint.v() * u_MAC.v()) / (myg.dx * self.r2v) + scalar_1.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (D v)_y
            (scalar_yint.jp(1) * (v_MAC.jp(1) +
             U0_half.jp(1)[np.newaxis,:]) -
             scalar_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy + 2. * scalar_1.v() * (v.v() + U0_half.v2d()) / self.r2v)

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
             Dh0_yint.v() * U0_half.v2d())/myg.dy + 2. * Dh0.v2df(myg.qx) * (v.v() + U0_half.v2d()) / self.r2v) + \
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
        Dh_2 = myg.scratch_array(data=Dh_1.d)
        drp0 = self.drp0(Dh0=Dh0, u=u_MAC, v=v_MAC, u0=u0_MAC)

        Dh_2.v()[:,:] += -self.dt * (
            #  (D u)_x
            (Dh_xint.ip(1)*u_MAC.ip(1) - Dh_xint.v()*u_MAC.v())(myg.dx * self.r2v) + Dh_1.v() * u.v() / (self.r2v * np.tan(myg.x2v)) +
            #  (D v)_y
            (Dh_yint.jp(1)*(v_MAC.jp(1) + U0_half.jp(1)[np.newaxis,:]) -\
             Dh_yint.v() * (v_MAC.v() + U0_half.v2d())) / myg.dy + 2. * Dh_1.v() * (v.v() + U0_half.v2d()) / self.r2v) + \
            self.dt * u0_MAC.v() * v_MAC.v() * drp0.v2d() + \
            self.dt * u0_MAC.v() * psi.v()

        self.enforce_tov(u=u_MAC, v=v_MAC, u0=u0_MAC)

        # update eint as a diagnostic
        eint = self.aux_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:,:] = self.base["p0"].v2d()/(gamma - 1.0)/D.v()

        # update T based on EoS
        T_2 = myg.scratch_array()
        self.calc_T(p0=p0, D=D_2, DX=DX_2, u=u_MAC, v=v_MAC, u0=u0_MAC, T=T_2)

        #---------------------------------------------------------------------
        # 9. React state through dt/2
        #---------------------------------------------------------------------
        D.d[:,:] = D_2.d[:,:]
        Dh.d[:,:] = Dh_2.d[:,:]
        DX.d[:,:] = DX_2.d[:,:]
        scalar.d[:,:] = scalar_2.d[:,:]
        T.d[:,:] = T_2.d[:,:]
        self.react_state(S=self.compute_S(u=u_MAC, v=v_MAC), D=D, Dh=Dh, DX=DX, T=T, scalar=scalar, u=u_MAC, v=v_MAC, u0=u0_MAC)

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("enthalpy")
        self.cc_data.fill_BC("scalar")

        #---------------------------------------------------------------------
        # 10. Define the new time expansion S and Gamma1
        #---------------------------------------------------------------------
        Q_2, _ = self.calc_Q_omega_dot(D=D_2, DX=DX_2, u=u_MAC, v=v_MAC, u0=u0_MAC, T=T_2)
        oldS = myg.scratch_array(data=S.d)

        S = self.compute_S(u=u_MAC, v=v_MAC, u0=u0_MAC, Q=Q_2, D=D_2)

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
        mg = rectMG.RectMG2d(myg.nx, myg.ny,
                             xl_BC_type=self.aux_data.BCs["phi"].xlb,
                             xr_BC_type=self.aux_data.BCs["phi"].xrb,
                             yl_BC_type=self.aux_data.BCs["phi"].ylb,
                             yr_BC_type=self.aux_data.BCs["phi"].yrb,
                             xmin=myg.xmin, xmax=myg.xmax,
                             ymin=myg.ymin, ymax=myg.ymax,
                             coeffs=coeff,
                             coeffs_bc=self.cc_data.BCs["density"],
                             verbose=0)

        # first compute div{zeta U}

        # u/v are cell-centered, divU is cell-centered
        # this bit seems to use U^n+1 rather than U_MAC
        div_zeta_U.v()[:,:] = \
            0.5 * zeta_half.v2d() * (u.ip(1) - u.ip(-1)) / (myg.dx * self.r2v) + \
            0.5 * (zeta_half.v2dp(1) * v.jp(1) - \
            zeta_half.v2dp(-1) * v.jp(-1)) / myg.dy + zeta_half.v2d() * u.v() / (self.r2v * np.tan(myg.x2v)) + zeta_half.v2d() * 2. * v.v() / self.r2v

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

        ###########################

        ##### GRADP USED HERE #####

        ###########################
        #gradp_x.d[:,:] = 1.
        #gradp_y.d[:,:] = 1.

        # FIXME: need base state forcing here!
        # However, it doesn't actually work: it just causes the atmosphere to rise up?
        base_forcing = self.base_state_forcing()
        # FIXME: This line messes up the velocity somehow - what on earth is it doing????
        #u.v()[:,:] += self.dt * (-coeff.v() * gradphi_x.v())
        # FIXME: add back in??
        #v.v()[:,:] += self.dt * (-coeff.v() * gradphi_y.v() + base_forcing.v())

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

        self.aux_data.fill_BC("gradp_x")
        self.aux_data.fill_BC("gradp_y")

        # FIXME: bcs for base state data
        for var in self.base.values():
            for gz in range(1,myg.ng):
                var.d[myg.jlo-gz] = var.d[myg.jlo]
                var.d[myg.jhi+gz] = var.d[myg.jhi]

        plot_me.d[:,:] =  D.d - D_old.d

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
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        DX = self.cc_data.get_var("mass-frac")
        scalar = self.cc_data.get_var("scalar")
        T = self.cc_data.get_var("temperature")

        #plot_me = self.aux_data.get_var("plot_me")

        myg = self.cc_data.grid

        psi = myg.scratch_array(data=scalar.d/D.d)
        X = myg.scratch_array(data=DX.d/D.d)
        logT = myg.scratch_array(data=np.log(T.d))

        magvel = np.sqrt(u**2 + v**2)

        vort = myg.scratch_array()

        dv = 0.5 * (v.ip(1) - v.ip(-1)) / (myg.dx * self.r2v) + v.v() / (self.r2v * np.tan(myg.x2v))
        du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy + 2. * u.v() / self.r2v

        vort.v()[:,:] = dv - du

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.3)

        fields = [D, X, u, logT]
        field_names = [r"$D$", r"$X$", r"$u$", r"$\ln T$"]
        colourmaps = [cmaps.magma_r, cmaps.magma, cmaps.viridis_r,
                      cmaps.magma]

        #vmaxes = [0.05, 1.0, 0.64, None]
        #vmins = [0.0, 0.95, 0.0, 3.0]

        for n in range(len(fields)):
            ax = axes.flat[n]

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
