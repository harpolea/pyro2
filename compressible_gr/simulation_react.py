from __future__ import print_function

#import sys

import numpy as np
import matplotlib.pyplot as plt
#import math

#import compressible_gr.BC as BC
from compressible_gr.simulation import *
from compressible_gr.problems import *
import compressible_gr.eos as eos
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
from compressible_gr.unsplitFluxes import *
from scipy.ndimage import median_filter
import compressible_gr.cons_to_prim as cy
from mesh.patch import ArrayIndexer

class SimulationReact(Simulation):

    def compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """
        myg = self.cc_data.grid

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        D = self.cc_data.get_var("D")
        Sx = self.cc_data.get_var("Sx")
        Sy = self.cc_data.get_var("Sy")
        tau = self.cc_data.get_var("tau")
        DX = self.cc_data.get_var("DX")

        gamma = self.rp.get_param("eos.gamma")
        c = self.rp.get_param("eos.c")
        u = np.zeros_like(D.d)
        v = np.zeros_like(D.d)
        rho = np.zeros_like(D.d)
        p = np.zeros_like(D.d)
        cs = np.zeros_like(D.d)

        U = myg.scratch_array(self.vars.nvar)
        U.d[:,:,self.vars.iD] = D.d
        U.d[:,:,self.vars.iSx] = Sx.d
        U.d[:,:,self.vars.iSy] = Sy.d
        U.d[:,:,self.vars.itau] = tau.d
        U.d[:,:,self.vars.iDX] = DX.d

        V = myg.scratch_array(self.vars.nvar)
        V.d[:,:,:] = cy.cons_to_prim(U.d, c, gamma, myg.qx, myg.qy, self.vars.nvar, self.vars.iD, self.vars.iSx, self.vars.iSy, self.vars.itau, self.vars.iDX)

        rho = ArrayIndexer(d=V.d[:,:,self.vars.irho], grid=myg)
        u = ArrayIndexer(d=V.d[:,:,self.vars.iu], grid=myg)
        v = ArrayIndexer(d=V.d[:,:,self.vars.iv], grid=myg)
        p = ArrayIndexer(d=V.d[:,:,self.vars.ip], grid=myg)
        X = ArrayIndexer(d=V.d[:,:,self.vars.iX], grid=myg)

        cs = sound_speed(gamma, rho.d, p.d)
        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        maxvel = np.fabs(np.sqrt(u.d**2 + v.d**2)).max()
        maxcs = cs.max()

        denom, _ = rel_add_velocity(maxvel, 0., maxcs, 0., c)

        xtmp = self.cc_data.grid.dx / denom
        ytmp = self.cc_data.grid.dy / denom

        T = self.calc_T(p, D, X, rho)

        # find burning stuff
        Q, omega_dot = self.calc_Q_omega_dot(D, X, rho, T)

        # stop them being too small so don't divide by zero
        Q.d[abs(Q.d) < 1.e-15] = 1.e-15
        omega_dot.d[abs(omega_dot.d) < 1.e-15] = 1.e-15

        burning_dt = cfl * min(1./np.max(abs(Q.d)), 1./np.max(abs(omega_dot.d)))

        # FIXME: get rid of 0.01
        #self.dt = min(burning_dt, 0.01 * cfl * min(xtmp.min(), ytmp.min()))
        self.dt = 0.1 * min(burning_dt, cfl * min(xtmp.min(), ytmp.min()))

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
        T9 = T.d * 1.e1
        rho5 = rho.d * 1.e3

        # make sure don't get divide by zero
        T.d[T.d < 1.e-15] = 1.e-15

        # don't apply to X > 1 as will get negative Q
        Q.d[X.d < 1.] = rho5[X.d < 1.]**2 * ((1.-X.d[X.d < 1.]) / T9[X.d < 1.])**3 * np.exp(-4.4 / T9[X.d < 1.])

        # Hnuc = |Delta q|omega_dot, where Delta q is the change in binding energy. q_He = 2.83007e4 keV, q_C=9.2161753e4 keV
        # sr_bubble:
        if self.problem_name == 'sr_bubble':
            omega_dot.d[:,:] = Q.d * 1.e3#* 9.773577e10
        elif self.problem_name == 'kh':
            omega_dot.d[:,:] = Q.d * 1.e11
        elif self.problem_name == 'sod':
            #Q.d[:,:] *= 2.
            omega_dot.d[:,:] = Q.d
        else:
            omega_dot.d[:,:] = Q.d

        # need to stop it getting bigger than one - this does this smoothly.
        #omega_dot.d[:,:] *= 0.5 * (1. - np.tanh(40. * (omega_dot.d - 1.)))

        # stop anything less than 0 so don't have reverse reactions.
        omega_dot.d[omega_dot.d < 0.] = 0.

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
        gamma = self.rp.get_param("eos.gamma")
        c = self.rp.get_param("eos.c")

        # get conserved and primitive variables.
        D = self.cc_data.get_var("D")
        Sx = self.cc_data.get_var("Sx")
        Sy = self.cc_data.get_var("Sy")
        tau = self.cc_data.get_var("tau")
        DX = self.cc_data.get_var("DX")

        U = myg.scratch_array(self.vars.nvar)
        U.d[:,:,self.vars.iD] = D.d
        U.d[:,:,self.vars.iSx] = Sx.d
        U.d[:,:,self.vars.iSy] = Sy.d
        U.d[:,:,self.vars.itau] = tau.d
        U.d[:,:,self.vars.iDX] = DX.d

        rho = myg.scratch_array()
        u = myg.scratch_array()
        v = myg.scratch_array()
        p = myg.scratch_array()
        X = myg.scratch_array()

        V = myg.scratch_array(self.vars.nvar)
        V.d[:,:,:] = cy.cons_to_prim(U.d, c, gamma, myg.qx, myg.qy, self.vars.nvar, self.vars.iD, self.vars.iSx, self.vars.iSy, self.vars.itau, self.vars.iDX)

        rho.d[:,:] = V.d[:,:,self.vars.irho]
        u.d[:,:] = V.d[:,:,self.vars.iu]
        v.d[:,:] = V.d[:,:,self.vars.iv]
        p.d[:,:] = V.d[:,:,self.vars.ip]
        X.d[:,:] = V.d[:,:,self.vars.iX]

        T = self.calc_T(p, D, X, rho)

        mp_kB = self.rp.get_param("eos.mp_kb")
        kB_mp = 1./mp_kB
        gamma = self.rp.get_param("eos.gamma")
        c = self.rp.get_param("eos.c")

        Q, omega_dot = self.calc_Q_omega_dot(D, X, rho, T)

        #h_T = kB_mp * gamma * (3. - 2. * X.d) / (6. * (gamma-1.))
        #h_X = -kB_mp * T.d * gamma / (3. * (gamma-1.))

        blank = D.d * 0.0

        w = 1. / np.sqrt(1. - (u.d**2 + v.d**2)/c**2)

        Sx_F = myg.scratch_array()
        Sy_F = myg.scratch_array()
        tau_F = myg.scratch_array()
        DX_F = myg.scratch_array()

        Sx_F.d[:,:] = D.d * Q.d * w * u.d
        Sy_F.d[:,:] = D.d * Q.d * w * v.d

        # negative energy flux as Q is like a potential energy that gets released when burning happens.
        tau_F.d[:,:] = -D.d * Q.d * w

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

        rho = myg.scratch_array()
        p = myg.scratch_array()
        X = myg.scratch_array()
        u = myg.scratch_array()
        v = myg.scratch_array()
        h = myg.scratch_array()
        S = myg.scratch_array()

        U = myg.scratch_array(self.vars.nvar)
        U.d[:,:,self.vars.iD] = D.d
        U.d[:,:,self.vars.iSx] = Sx.d
        U.d[:,:,self.vars.iSy] = Sy.d
        U.d[:,:,self.vars.itau] = tau.d
        U.d[:,:,self.vars.iDX] = DX.d

        V = myg.scratch_array(self.vars.nvar)
        V.d[:,:,:] = cy.cons_to_prim(U.d, c, gamma, myg.qx, myg.qy, self.vars.nvar, self.vars.iD, self.vars.iSx, self.vars.iSy, self.vars.itau, self.vars.iDX)

        rho.d[:,:] = V.d[:,:,self.vars.irho]
        u.d[:,:] = V.d[:,:,self.vars.iu]
        v.d[:,:] = V.d[:,:,self.vars.iv]
        p.d[:,:] = V.d[:,:,self.vars.ip]
        h.d[:,:] = V.d[:,:,self.vars.itau]
        X.d[:,:] = V.d[:,:,self.vars.iX]

        # get the velocity magnitude
        magvel = myg.scratch_array()
        magvel.d[:,:] = np.sqrt(u.d**2 + v.d**2)

        def discrete_Laplacian(f):
            return (f.ip(1) - 2.*f.v() + f.ip(-1)) / myg.dx**2 + \
                   (f.jp(1) - 2.*f.v() + f.jp(-1)) / myg.dy**2

        T = self.calc_T(p, D, X, rho)
        #T.d[:,:] = np.log(T.d)

        vort = myg.scratch_array()

        dv = 0.5 * (v.ip(1) - v.ip(-1)) / myg.dx
        du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy

        vort.v()[:,:] = dv - du

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.

        # figure out the geometry
        L_x = myg.xmax - myg.xmin
        L_y = myg.ymax - myg.ymin

        orientation = "vertical"
        shrink = 1.0

        sparseX = 0
        allYlabel = 1

        # BRITGRAV PLOT
        img = plt.imshow(np.transpose(X.v()),
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax], vmin=0, vmax=1, cmap=plt.get_cmap('viridis'))
        ax = plt.axes()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tight_layout()

"""
        if L_x >= 2*L_y:

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

            onLeft = [0, 2]

        xcntr = np.round(0.5 * myg.qx).astype(int)
        ycntr = np.round(0.5 * myg.qy).astype(int)

        if self.problem_name == 'kh':
            Q, omega_dot = self.calc_Q_omega_dot(D, X, rho, T)
            fields = [rho, omega_dot, X, vort]
            field_names = [r"$\rho$", r"$\dot{\omega}$", r"$X$", r"$\nabla\times u$"]
            colourmaps = [plt.get_cmap('viridis'), plt.get_cmap('viridis'), plt.get_cmap('viridis'),  plt.get_cmap('viridis')]

        elif self.problem_name == 'sr_bubble':
            # Schlieren
            S.v()[:,:] = np.log(abs(discrete_Laplacian(rho)))
            # low pass and median filters to clean up plot
            S.d[S.d < -5] = -6.
            S.d[:,:] = median_filter(S.d, 4)

            fields = [rho, omega_dot, X, S]
            field_names = [r"$\rho$", r"$\dot{\omega}$", r"$X$", r"$\ln|\mathcal{S}|$"]
            colourmaps = [plt.get_cmap('viridis'), plt.get_cmap('viridis'), plt.get_cmap('viridis'),  plt.get_cmap('Greys')]

        elif self.problem_name == 'sod':
            Q, omega_dot = self.calc_Q_omega_dot(D, X, rho, T)
            #fields = [rho, T, X, omega_dot]
            #field_names = [r"$\rho$", r"$T$", r"$X$", r"$\dot{\omega}$"]
            fields = [rho, u, X, p]
            field_names = [r"$\rho$", r"$u$", r"$X$", r"$p$"]
            colourmaps = [plt.get_cmap('viridis'), plt.get_cmap('viridis'), plt.get_cmap('viridis'),  plt.get_cmap('viridis')]

        else:
            Q, omega_dot = self.calc_Q_omega_dot(D, X, rho, T)
            fields = [rho, omega_dot, X, vort]
            field_names = [r"$\rho$", r"$\dot{\omega}$", r"$X$", r"$\nabla\times u$"]
            colourmaps = [plt.get_cmap('viridis'), plt.get_cmap('viridis'), plt.get_cmap('viridis'),  plt.get_cmap('viridis')]

        # colours of line plots
        colours = ['blue', 'red', 'black', 'green']

        for n in range(4):
            ax = axes.flat[2*n]
            ax2 = axes.flat[2*n+1]

            v = fields[n]
            img = ax.imshow(np.transpose(v.v()),
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax], vmin=vmins[n], vmax=vmaxes[n], cmap=colourmaps[n])

            # line plots
            if self.problem_name == 'kh':
                # shall do a slice vertically rather than horizontally here.
                x2 = myg.y
                y2 = v.d[xcntr,:]
            elif self.problem_name == 'sr_bubble':
                x2 = myg.x
                y2 = v.d[:,ycntr]
            else:
                x2 = myg.x
                y2 = v.d[:,ycntr]

            plt2 = ax2.plot(x2, y2, c=colours[n])
            ax2.set_xlim([min(x2), max(x2)])

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
"""
