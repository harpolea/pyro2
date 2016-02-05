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
#import multigrid.rect_MG as rectMG
from util import profile
import mesh.metric as metric
import colormaps as cmaps

class SimulationReact(Simulation):

    def __init__(self, solver_name, problem_name, rp, timers=None, fortran=True, testing=False):
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

        Simulation.__init__(self, solver_name, problem_name, rp, timers=timers, fortran=fortran, testing=testing)

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
            u0 = myg.metric.calcu0(u=u, v=v)
        if p0 is None:
            p0 = self.base["p0"]
        if Q is None:
            Q = myg.scratch_array()
        if D is None:
            D = self.cc_data.get_var("density")
        gamma = self.rp.get_param("eos.gamma")

        chrls = myg.metric.chrls

        S.d[:,:] = super(SimulationReact, self).compute_S(u=u, v=v, u0=u0, p0=p0).d

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
            u0 = myg.metric.calcu0(u=u, v=v)
        if T is None:
            T = self.cc_data.get_var("temperature")

        # mean molecular weight = (2*(1-X) + 3/4 X)^-1
        mu = 4./(8. * (1. - DX.d/D.d) + 3. * DX.d/D.d)
        # FIXME: hack to drive reactions
        mp_kB = 1.21147#e5#e-8

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
            u0 = myg.metric.calcu0(u=u, v=v)
        if T is None:
            T = self.cc_data.get_var("temperature")
        Q = myg.scratch_array()
        omega_dot = myg.scratch_array()
        # FIXME: hack to drive reactions
        T9 = T.d #* 1.e-9#1.e-9
        D5 = D.d #* 1.e-5

        Q.d[:,:] = 5.3 * (D5 / u0.d)**2 * ((1.-DX.d/D.d) * DX.d/D.d / T9)**3 * np.exp(-4.4 / T9)

        #print((np.exp(-4.4 / T9))[25:35,25:35])

        # Hnuc = |Delta q|omega_dot, where Delta q is the change in binding energy. q_He = 2.83007e4 keV, q_C=9.2161753e4 keV
        omega_dot.d[:,:] = Q.d #* 9.773577e10

        # FIXME: hackkkkk
        if self.problem_name == 'bubble':
            Q.d[:,:] *= 1.e9
            omega_dot.d[:,:] *= 5.
        else:
            Q.d[:,:] *= 1.e12
            omega_dot.d[:,:] *= 1.e4

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
            u0 = myg.metric.calcu0(u=u, v=v)
        drp0 = self.drp0(Dh0=Dh0, u=u, v=v, u0=u0)
        if S is None:
            S = self.aux_data.get_var("source_y")
        if T is None:
            T = self.cc_data.get_var("temperature")
        kB_mp = 8.254409#e7
        gamma = self.rp.get_param("eos.gamma")

        Q, omega_dot = self.calc_Q_omega_dot(D=D, DX=DX, u=u, v=v, u0=u0, T=T)

        h_T = kB_mp * gamma * (3. - 2. * DX.d/D.d) / (6. * (gamma-1.))
        h_X = -kB_mp * T.d * gamma / (3. * (gamma-1.))

        # cannot call super here as it changes D which is then needed to calculate Dh, D

        Dh.d[:,:] += 0.5 * self.dt * (S.d * Dh.d + u0.d * v.d * drp0.d2d() + D.d * Q.d)

        DX.d[:,:] += 0.5 * self.dt * (S.d * DX.d + D.d * omega_dot.d)

        D.d[:,:] += 0.5 * self.dt * (S.d * D.d)

        scalar.d[:,:] += 0.5 * self.dt * (S.d * scalar.d)

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
