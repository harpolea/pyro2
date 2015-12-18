from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt
import math

import compressible_gr.BC as BC
from compressible_gr.simulation import *
from compressible_gr.problems import *
import compressible_gr.eos as eos
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
from compressible_gr.unsplitFluxes import *
from util import profile
import lm_gr.metric as metric
import colormaps as cmaps
from compressible_gr.unsplitFluxes import *

class SimulationReact(Simulation):

    def calc_T(self, p, D, X, rho):
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
        p : ArrayIndexer object
            pressure
        D : ArrayIndexer object
            conservative density
        X : ArrayIndexer object
            mass fraction
        rho : ArrayIndexer object
            primitive density
        T : ArrayIndexer object
            temperature
        """
        myg = self.cc_data.grid
        mp_kB = self.rp.get_param("eos.mp_kb")

        T = myg.scratch_array()

        # mean molecular weight = (2*(1-X) + 3/4 X)^-1
        mu = 4. / (8. * (1. - X.d) + 3. * X.d)

        T.d[:,:] = p.d * mu * mp_kB / rho.d

        return T


    def calc_Q_omega_dot(self, D, X, rho, T):
        r"""
        Calculates the energy generation rate according to eq. 2 of Cavecchi, Levin, Watts et al 2015 and the creation rate

        Parameters
        ----------
        D : ArrayIndexer object
            conservative density
        X : ArrayIndexer object
            mass fraction
        rho : ArrayIndexer object
            primitive density
        T : ArrayIndexer object
            temperature

        Returns
        -------
        Q : ArrayIndexer object
            nuclear heating rate
        omega_dot : ArrayIndexer object
            species creation rate
        """
        myg = self.cc_data.grid

        Q = myg.scratch_array()
        omega_dot = myg.scratch_array()
        # FIXME: hack to drive reactions
        T9 = T.d #* 1.e-9#1.e-9
        rho5 = rho.d #* 1.e-5

        Q.d[:,:] = 5.3 * rho5**2 * (X.d / T9)**3 * np.exp(-4.4 / T9)

        #print((np.exp(-4.4 / T9))[25:35,25:35])

        # Hnuc = |Delta q|omega_dot, where Delta q is the change in binding energy. q_He = 2.83007e4 keV, q_C=9.2161753e4 keV
        omega_dot.d[:,:] = Q.d #* 9.773577e10
        # need to stop it getting bigger than one - this does this smoothly.
        omega_dot.d[:,:] *= 0.5 * (1. - np.tanh(40. * (omega_dot.d - 1.)))

        # FIXME: hackkkkk
        #Q.d[:,:] *= 1.e12 # for bubble: 1.e9, else 1.e12
        #omega_dot.d[:,:] *= 5. # for bubble: 5., else 1.e4
        #print(omega_dot.d[20:30, 8:12])

        return Q, omega_dot

    def burning_flux(self):
        """
        burning source terms

        Parameters
        ----------
        D : ArrayIndexer object
            conservative density
        h : ArrayIndexer object
            enthalpy
        DX : ArrayIndexer object
            conservative density * species mass fraction
        T : ArrayIndexer object
            temperature
        scalar : ArrayIndexer object
            passive advective scalar (* density)
        rho : ArrayIndexer object
            primitive density
        """

        myg = self.cc_data.grid

        # get conserved and primitive variables.
        D = self.cc_data.get_var("D")
        Sx = self.cc_data.get_var("Sx")
        Sy = self.cc_data.get_var("Sy")
        tau = self.cc_data.get_var("tau")
        DX = self.cc_data.get_var("DX")

        gamma = self.rp.get_param("eos.gamma")
        c = self.rp.get_param("eos.c")
        _rho = np.zeros_like(D.d)
        _u = np.zeros_like(D.d)
        _v = np.zeros_like(D.d)
        _p = np.zeros_like(D.d)
        _X = np.zeros_like(D.d)

        # we need to compute the primitive speeds and sound speed
        for i in range(myg.qx):
            for j in range(myg.qy):
                U = (D.d[i,j], Sx.d[i,j], Sy.d[i,j], tau.d[i,j], DX.d[i,j])
                names = ['D', 'Sx', 'Sy', 'tau', 'DX']
                nan_check(U, names)
                V, _ = cons_to_prim(U, c, gamma)

                _rho[i,j], _u[i,j], _v[i,j], _, _p[i,j], _X[i,j] = V

        rho = myg.scratch_array()
        u = myg.scratch_array()
        v = myg.scratch_array()
        p = myg.scratch_array()
        X = myg.scratch_array()
        rho.d[:,:] = _rho
        u.d[:,:] = _u
        v.d[:,:] = _v
        p.d[:,:] = _p
        X.d[:,:] = _X

        T = self.calc_T(p, D, X, rho)

        mp_kB = self.rp.get_param("eos.mp_kb")
        kB_mp = 1./mp_kB
        gamma = self.rp.get_param("eos.gamma")
        c = self.rp.get_param("eos.c")

        Q, omega_dot = self.calc_Q_omega_dot(D, X, rho, T)

        h_T = kB_mp * gamma * (3. - 2. * X.d) / (6. * (gamma-1.))
        h_X = -kB_mp * T.d * gamma / (3. * (gamma-1.))

        blank = D.d * 0.0

        w = 1 / np.sqrt(1 - (u.d**2 + v.d**2)/c**2)

        Sx_F = myg.scratch_array()
        Sy_F = myg.scratch_array()
        tau_F = myg.scratch_array()
        DX_F = myg.scratch_array()

        Sx_F.d[:,:] = D.d * Q.d * w * u.d
        Sy_F.d[:,:] = D.d * Q.d * w * v.d

        tau_F.d[:,:] = D.d * Q.d * w

        DX_F.d[:,:] = D.d * omega_dot.d

        #T.d[:,:] += self.dt * (Q.d - h_X * omega_dot.d) / h_T

        return (blank, Sx_F, Sy_F, tau_F, DX_F)


    def dovis(self, vmins=[None, None, None, None], vmaxes=[None, None, None, None]):
        """
        Do runtime visualization
        """

        plt.clf()

        plt.rc("font", size=12)
        myg = self.cc_data.grid

        D = self.cc_data.get_var("D")
        DX = self.cc_data.get_var("DX")
        Sx = self.cc_data.get_var("Sx")
        Sy = self.cc_data.get_var("Sy")
        tau = self.cc_data.get_var("tau")

        gamma = self.cc_data.get_aux("gamma")
        c = self.cc_data.get_aux("c")
        u = np.zeros_like(D.d)
        v = np.zeros_like(D.d)

        rho = myg.scratch_array()
        p = myg.scratch_array()
        h = myg.scratch_array()
        X = myg.scratch_array()
        _u = myg.scratch_array()

        for i in range(myg.qx):
            for j in range(myg.qy):
                F = (D.d[i,j], Sx.d[i,j], Sy.d[i,j], tau.d[i,j], DX.d[i,j])
                Fp, cs = cons_to_prim(F, c, gamma)
                rho.d[i,j], u[i,j], v[i,j], h.d[i,j], p.d[i,j], X.d[i,j] = Fp

        # get the pressure
        magvel = myg.scratch_array()
        magvel.d[:,:] = np.sqrt(u**2 + v**2)

        T = self.calc_T(p, D, X, rho)
        T.d[:,:] = np.log(T.d)
        _u.d[:,:] = u

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.

        # figure out the geometry
        L_x = myg.xmax - myg.xmin
        L_y = myg.ymax - myg.ymin

        orientation = "vertical"
        shrink = 1.0

        sparseX = 0
        allYlabel = 1

        if L_x > 2*L_y:

            # we want 4 rows:
            #  rho
            #  |U|
            #   p
            #   e
            fig, axes = plt.subplots(nrows=4, ncols=2, num=1)
            orientation = "horizontal"
            if (L_x > 4.*L_y):
                shrink = 0.75

            onLeft = list(range(self.vars.nvar))


        elif L_y > 2.*L_x:

            # we want 4 columns:  rho  |U|  p  e
            fig, axes = plt.subplots(nrows=1, ncols=4, num=1)
            if (L_y >= 3.*L_x):
                shrink = 0.5
                sparseX = 1
                allYlabel = 0

            onLeft = [0]

        else:
            # 2x2 grid of plots with
            #
            #   rho   |u|
            #    p     e
            fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
            plt.subplots_adjust(hspace=0.25)

            onLeft = [0,2]


        fields = [rho, _u, T, X]
        field_names = [r"$\rho$", r"$u$", "$\ln(T)$", "$X$"]
        colours = ['blue', 'red', 'black', 'green']

        for n in range(4):
            ax = axes.flat[2*n]
            ax2 = axes.flat[2*n+1]

            v = fields[n]
            ycntr = np.round(0.5 * myg.qy).astype(int)
            img = ax.imshow(np.transpose(v.v()),
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax], vmin=vmins[n], vmax=vmaxes[n])
            plt2 = ax2.plot(myg.x, v.d[:,ycntr], c=colours[n])
            ax2.set_xlim([myg.xmin, myg.xmax])


            #ax.set_xlabel("x")
            if n==3:
                ax2.set_xlabel("$x$")
            if n == 0:
                ax.set_ylabel("$y$")
                ax2.set_ylabel(field_names[n], rotation='horizontal')
            elif allYlabel:
                ax.set_ylabel("$y$")
                ax2.set_ylabel(field_names[n], rotation='horizontal')

            ax.set_title(field_names[n])

            if not n in onLeft:
                ax.yaxis.offsetText.set_visible(False)
                ax2.yaxis.offsetText.set_visible(False)
                if n > 0:
                    ax.get_yaxis().set_visible(False)
                    ax2.get_yaxis().set_visible(False)

            if sparseX:
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax2.xaxis.set_major_locator(plt.MaxNLocator(3))

            ax2.set_ylim([vmins[n], vmaxes[n]])
            plt.colorbar(img, ax=ax, orientation=orientation, shrink=0.75)


        plt.figtext(0.05,0.0125, "n: %4d,   t = %10.5f" % (self.n, self.cc_data.t))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.4, wspace=0.1)
        #plt.tight_layout()

        #plt.draw()
