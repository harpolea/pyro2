"""
TODO: MAKE IT POSSIBLE TO START A PROGRAM FROM EXISTING OUTPUT FILE?
e.g. maybe do a dump of EVERYTHING at the end of the program so that it's possible to read this in and set up the problem in the state it left off.

TODO: D ln u0/Dt term in momentum equation?

FIXME: check edge/cell-centred/time-centred quantities used correctly

CHANGED: moved tov update in steps 4 and 8 to after the Dh0 update as it is a
function of Dh0

TODO: not entirely sure about the lateral averaging of D, Dh to get the base states in steps 4 and 8

CHANGED: made python3 compliable (for the most part - f2py doesn't seem to be able to work too well with python3)

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
from lm_gr.simulation import *
import multigrid.variable_coeff_MG as vcMG
from util import profile
import lm_gr.metric as metric
import colormaps as cmaps

class SimulationReact(Simulation):

    def __init__(self, solver_name, problem_name, rp, timers=None, fortran=True):
        """
        Initialize the SimulationReact object

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

        Simulation.__init__(self, solver_name, problem_name, rp, timers=timers, fortran=fortran)


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
        #my_data.register_var("phi-MAC", bc_dens) # moved to aux_data

        # mass fraction: starts as zero, burns to 1
        my_data.register_var("mass-frac", bc_dens)

        # passive scalar that is advected along - this is actually going to be the scalar times the density, so be careful when plotting at the end
        my_data.register_var("scalar", bc_dens)

        # temperature
        my_data.register_var("temperature", bc_dens)

        my_data.create()

        self.cc_data = my_data

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = patch.CellCenterData2d(myg)

        # we'll keep the internal energy around just as a diagnostic
        aux_data.register_var("eint", bc_dens)

        aux_data.register_var("coeff", bc_dens)
        aux_data.register_var("source_y", bc_yodd)
        aux_data.register_var("old_source_y", bc_yodd)
        aux_data.register_var("plot_me", bc_yodd) # somewhere to store data for plotting
        aux_data.register_var("phi", bc_dens)
        aux_data.register_var("phi-MAC", bc_dens)

        # gradp -- used in the projection and interface states.  We'll do the
        # same BCs as density
        aux_data.register_var("gradp_x", bc_dens)
        aux_data.register_var("gradp_y", bc_dens)

        aux_data.create()
        self.aux_data = aux_data

        #  1-d base state
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
        exec(self.problem_name + '.init_data(self.cc_data, self.aux_data, self.base, self.rp, self.metric)')

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


    def compute_S(self, u=None, v=None, u0=None, p0=None, Q=None, D=None):
        r"""
        :math: `S = -\Gamma^\mu_(\mu \nu) U^\nu + \frac{D (\gamma-1)}{u^0 \gamma^2 p} H`  (see eq 6.34, 6.37 in LowMachGR).
        base["source-y"] is not updated here as it's sometimes necessary to
        calculate projections of `S` (e.g. `S^{n*}`) and not `S^n`

        Parameters
        ----------
        u : ArrayIndexer object, optional
            horizontal velocity
        v : ArrayIndexer object, optional
            vertical velocity
        u0 : ArrayIndexer object, optional
            timelike component of the 4-velocity
        p0 : ArrayIndexer object, optional
            base-state pressure
        Q : ArrayIndexer object, optional
            nuclear heating rate
        D : ArrayIndexer object, optional
            full density state

        Returns
        -------
        S : ArrayIndexer object
        """
        myg = self.cc_data.grid
        S = myg.scratch_array()
        if u is None:
            u = self.cc_data.get_var("x-velocity")
        if v is None:
            v = self.cc_data.get_var("y-velocity")
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        if p0 is None:
            p0 = self.base["p0"]
        if Q is None:
            Q = myg.scratch_array()
        if D is None:
            D = self.cc_data.get_var("density")
        gamma = self.rp.get_param("eos.gamma")

        #chrls = np.array([[self.metric.christoffels([self.cc_data.t, i, j])
        #                   for j in range(myg.qy)] for i in range(myg.qx)])
        # time-independent metric
        chrls = self.metric.chrls

        #S.d[:,:] = -(chrls[:,:,0,0,0] + chrls[:,:,1,1,0] + chrls[:,:,2,2,0] +
        #    (chrls[:,:,0,0,1] + chrls[:,:,1,1,1] + chrls[:,:,2,2,1]) * u.d +
        #    (chrls[:,:,0,0,2] + chrls[:,:,1,1,2] + chrls[:,:,2,2,2]) * v.d) + \
        #    D.d * (gamma - 1) * Q.d / (u0.d * gamma**2 * p0.d2d())

        S.d[:,:] = super().compute_S(u=u, v=v, u0=u0, p0=p0, D=D).d

        S.d[:,:] += D.d * (gamma - 1) * Q.d / (u0.d * gamma**2 * p0.d2d())

        return S


    def calc_T(self, p0=None, D=None, DX=None, u=None, v=None, u0=None, T=None):
        r"""
        Calculates the temperature assuming an ideal gas with a mixed composition and a constant ratio of specific heats:
        .. math::

            p = \rho e (\gamma -1) = \frac{\rho k_B T}{\mu m_p}

        The mean molecular weight, :math: `\mu`, is given by
        .. math::

            \mu = \left(\sum_k \frac{X_k}{A_k}\right)^{-1}

        We model the system as being made purely of helium which burns into carbon. X is the fraction of carbon, so we can write
        .. math::

            \mu = (2+4X)^{-1}

        Parameters
        ----------
        p0 : ArrayIndexer object, optional
            base state pressure
        D : ArrayIndexer object, optional
            full state density
        DX : ArrayIndexer object, optional
            density * mass fraction full state
        u : ArrayIndexer object, optional
            horizontal velocity
        v : ArrayIndexer object, optional
            vertical velocity
        u0 : ArrayIndexer object, optional
            timelike component of the 4-velocity
        T : ArrayIndexer object, optional
            temperature
        """
        myg = self.cc_data.grid
        if p0 is None:
            p0 = self.base["p0"]
        if D is None:
            D = self.cc_data.get_var("density")
        if DX is None:
            DX = self.cc_data.get_var("mass-frac")
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        if T is None:
            T = self.cc_data.get_var("temperature")

        # mean molecular weight
        mu = 1./(2. + 4. * DX.d/D.d)
        # FIXME: hack to drive reactions
        mp_kB = 1.21147#e-8

        # use p0 here as otherwise have to explicitly calculate pi somewhere?
        # TODO: could instead calculate this using Dh rather than p0?
        T.d[:,:] = p0.d2d() * mu * u0.d * mp_kB / D.d


    def calc_Q_omega_dot(self, D=None, DX=None, u=None, v=None, u0=None, T=None):
        r"""
        Calculates the energy generation rate according to eq. 2 of Cavecchi, Levin, Watts et al 2015 and the creation rate

        Parameters
        ----------
        D : ArrayIndexer object, optional
            full state density
        DX : ArrayIndexer object, optional
            density * mass fraction full state
        u : ArrayIndexer object, optional
            horizontal velocity
        v : ArrayIndexer object, optional
            vertical velocity
        u0 : ArrayIndexer object, optional
            timelike component of the 4-velocity
        T : ArrayIndexer object, optional
            temperature

        Returns
        -------
        Q : ArrayIndexer object
            nuclear heating rate
        omega_dot : ArrayIndexer object
            species creation rate
        """
        myg = self.cc_data.grid
        if D is None:
            D = self.cc_data.get_var("density")
        if DX is None:
            DX = self.cc_data.get_var("mass-frac")
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        if T is None:
            T = self.cc_data.get_var("temperature")
        Q = myg.scratch_array()
        omega_dot = myg.scratch_array()
        # FIXME: hack to drive reactions
        T9 = T.d * 1.6e-2#1.e-9 # for bubble: 1.45e-2
        D5 = D.d * 1.e-5

        Q.d[:,:] = 5.3e18 * (D5 / u0.d)**2 * ((1. - DX.d/D.d) / T9)**3 * np.exp(-4.4 / T9)

        # Hnuc = |Delta q|omega_dot, where Delta q is the change in binding energy. q_He = 2.83007e4 keV, q_C=9.2161753e4 keV
        omega_dot.d[:,:] = Q.d * 9.773577e10

        # FIXME: hackkkkk
        Q.d[:,:] *= 1.e12 # for bubble: 1.e9
        omega_dot.d[:,:] *= 1.e5 # for bubble: 5.

        return Q, omega_dot


    def react_state(self, S=None, D=None, Dh=None, DX=None, p0=None, T=None, scalar=None, Dh0=None, u=None, v=None, u0=None):
        """
        gravitational source terms in the continuity equation (called react
        state to mirror MAESTRO as here they just have source terms from the
        reactions)

        Parameters
        ----------
        S : ArrayIndexer object, optional
            source term
        D : ArrayIndexer object, optional
            density full state
        Dh : ArrayIndexer object, optional
            density * enthalpy full state
        DX : ArrayIndexer object, optional
            density * species mass fraction full state
        p0 : ArrayIndexer object, optional
            base state pressure
        T : ArrayIndexer object, optional
            temperature
        scalar : ArrayIndexer object, optional
            passive advective scalar (* density)
        Dh0 : ArrayIndexer object, optional
            density * enthalpy base state
        u : ArrayIndexer object, optional
            horizontal velocity
        v : ArrayIndexer object, optional
            vertical velocity
        u0 : ArrayIndexer object, optional
            timelike component of the 4-velocity
        """
        myg = self.cc_data.grid

        if D is None:
            D = self.cc_data.get_var("density")
        if Dh is None:
            Dh = self.cc_data.get_var("enthalpy")
        if DX is None:
            DX = self.cc_data.get_var("mass-frac")
        if scalar is None:
            scalar = self.cc_data.get_var("scalar")
        if v is None:
            v = self.cc_data.get_var("y-velocity")
        if u0 is None:
            u0 = self.metric.calcu0(u=u, v=v)
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v, u0=u0)
        if S is None:
            S = self.aux_data.get_var("source_y")
        if T is None:
            T = self.cc_data.get_var("temperature")
        # FIXME: hack to drive reactions
        kB_mp = 8.254409#e7
        gamma = self.rp.get_param("eos.gamma")

        Q, omega_dot = self.calc_Q_omega_dot(D=D, DX=DX, u=u, v=v, u0=u0, T=T)
        # print(Q.d.max())

        super().react_state(S=S, D=D, Dh=Dh, p0=p0, scalar=scalar, Dh0=Dh0, u=u, v=v, u0=u0)

        h_T = kB_mp * gamma * (3. - 2. * DX.d/D.d) / (6. * (gamma-1.))
        h_X = -kB_mp * T.d * gamma / (3. * (gamma-1.))

        Dh.d[:,:] += 0.5 * self.dt * (D.d * Q.d)

        DX.d[:,:] += 0.5 * self.dt * (S.d * DX.d + D.d * omega_dot.d)

        T.d[:,:] += 0.5 * self.dt * (Q.d - h_X * omega_dot.d) / h_T


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

        dv = 0.5 * (v.ip(1) - v.ip(-1)) / myg.dx
        du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy

        vort.v()[:,:] = dv - du

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.3)

        fields = [D, X, psi, logT]
        field_names = [r"$D$", r"$X$", r"$\psi$", r"$\ln T$"]
        colourmaps = [cmaps.magma_r, cmaps.magma, cmaps.viridis_r,
                      cmaps.magma]

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
