from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import compressible_gr.BC as BC
from compressible_gr.problems import *
import compressible_gr.eos as eos
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
from compressible_gr.unsplitFluxes import *
from util import profile


class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, iD=0, iSx=1, iSy=2, itau=3, iDX=4):
        self.nvar = 5

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.iD = iD
        self.iSx = iSx
        self.iSy = iSy
        self.itau = itau
        self.iDX = iDX

        # primitive variables
        self.irho = iD
        self.iu = iSx
        self.iv = iSy
        self.ip = itau
        self.iX = iDX
        #self.ih = 5


class Simulation(NullSimulation):

    def __init__(self, solver_name, problem_name, rp, timers=None):
        """
        Initialize the Simulation object

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
        """

        NullSimulation.__init__(self, solver_name, problem_name, rp, timers=timers)

        self.vars = Variables()

    def initialize(self):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)
        my_data = patch.CellCenterData2d(my_grid)

        # define solver specific boundary condition routines
        patch.define_bc("hse", BC.user)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # density and energy
        # order in which these are registers determines their variable indices
        my_data.register_var("D", bc)
        my_data.register_var("Sx", bc_xodd)
        my_data.register_var("Sy", bc_yodd)
        my_data.register_var("tau", bc)
        my_data.register_var("DX", bc)


        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("c", self.rp.get_param("eos.c"))
        my_data.set_aux("grav", self.rp.get_param("compressible-gr.grav"))

        my_data.create()

        self.cc_data = my_data

        self.vars = Variables(iD = my_data.vars.index("D"),
                              iSx = my_data.vars.index("Sx"),
                              iSy = my_data.vars.index("Sy"),
                              itau = my_data.vars.index("tau"),
                              iDX = my_data.vars.index("DX"))


        # initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')

        self.cc_data.fill_BC_all()

        if self.verbose > 0:
            print(my_data)


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
        cs = np.zeros_like(D.d)

        # we need to compute the primitive speeds and sound speed
        for i in range(myg.qx):
            for j in range(myg.qy):
                U = (D.d[i,j], Sx.d[i,j], Sy.d[i,j], tau.d[i,j], DX.d[i,j])
                names = ['D', 'Sx', 'Sy', 'tau', 'DX']
                # U here is wrong
                nan_check(U, names)
                V, cs[i,j] = cons_to_prim(U, c, gamma)

                _, u[i,j], v[i,j], _, _, _ = V

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        maxvel = np.fabs(np.sqrt(u**2 + v**2)).max()
        maxcs = cs.max()

        denom, _ = rel_add_velocity(maxvel, 0., maxcs, 0., c)

        xtmp = self.cc_data.grid.dx / denom
        ytmp = self.cc_data.grid.dy / denom

        self.dt = cfl * min(xtmp.min(), ytmp.min())

    # blank functions to be overriden by simulation_react
    def calc_T(self, p, D, DX, rho):
        return self.cc_data.grid.scratch_array()

    def calc_Q_omega_dot(self, D, DX, rho, T):
        return self.cc_data.grid.scratch_array(), self.cc_data.grid.scratch_array()

    def burning_flux(self):
        return (self.cc_data.grid.scratch_array() for n in range(self.vars.nvar))

    def smooth_lhs(self):
        """
        Smooths the state left of the shock to get rid of numerical
        artifacts just before the wave collides with the bubble.
        """
        print('Smoothing left hand side variables')
        x_pert = self.rp.get_param("sr-bubble.x_pert")
        r = self.rp.get_param("sr-bubble.r_pert")

        # location of bubble lhs edge
        x_c = x_pert - r

        myg = self.cc_data.grid

        smooth_mask = (myg.x - myg.xmin) > 0.1 * (x_c - myg.xmin)
        smooth_mask *= (myg.x - myg.xmin) < 0.98 * (x_c - myg.xmin)

        avg_mask = (myg.x - myg.xmin) < 0.1 * (x_c - myg.xmin)

        for n in range(self.vars.nvar):
            var = self.cc_data.get_var_by_index(n)
            # NOTE: might have mixed up axis here
            avg = np.mean(var.d[avg_mask], axis=0)
            var.d[smooth_mask] = avg[np.newaxis,:]


    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        grav = self.rp.get_param("compressible-gr.grav")

        myg = self.cc_data.grid

        burning_source = self.burning_flux()

        Flux_x, Flux_y = unsplitFluxes(self.cc_data, self.rp, self.vars, self.tc, self.dt, burning_source)


        for i in range(myg.qx):
            for j in range(myg.qy):
                nan_check(Flux_x.d[i,j,:], ['fx0', 'fx1', 'fx2', 'fx3'])
                nan_check(Flux_y.d[i,j,:], ['fy0', 'fy1', 'fy2', 'fy3'])

        # conservative update
        dtdx = self.dt / myg.dx
        dtdy = self.dt / myg.dy

        for n in range(self.vars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var.v()[:,:] += \
                dtdx * (Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy * (Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # gravitational source terms
        #Sy.d[:,:] += 0.5*self.dt*(D.d[:,:] + old_D.d[:,:])*grav
        #tau.d[:,:] += 0.5*self.dt*(Sy.d[:,:] + old_Sy.d[:,:])*grav

        self.cc_data.fill_BC_all()

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()


    def dovis(self, vmins=[None, None, None, None], vmaxes=[None, None, None, None]):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=12)
        myg = self.cc_data.grid

        D = self.cc_data.get_var("D")
        Sx = self.cc_data.get_var("Sx")
        Sy = self.cc_data.get_var("Sy")
        tau = self.cc_data.get_var("tau")
        DX = self.cc_data.get_var("DX")

        gamma = self.cc_data.get_aux("gamma")
        c = self.cc_data.get_aux("c")
        u = np.zeros_like(D.d)
        v = np.zeros_like(D.d)

        rho = myg.scratch_array()
        p = myg.scratch_array()
        h = myg.scratch_array()
        S = myg.scratch_array()

        for i in range(myg.qx):
            for j in range(myg.qy):
                F = (D.d[i,j], Sx.d[i,j], Sy.d[i,j], tau.d[i,j], DX.d[i,j])
                Fp, cs = cons_to_prim(F, c, gamma)
                rho.d[i,j], u[i,j], v[i,j], h.d[i,j], p.d[i,j], _ = Fp

        # get the velocity magnitude
        magvel = myg.scratch_array()
        magvel.d[:,:] = np.sqrt(u**2 + v**2)

        def discrete_Laplacian(f):
            return (f.ip(1) - 2.*f.v() + f.ip(-1)) / myg.dx**2 + \
                   (f.jp(1) - 2.*f.v() + f.jp(-1)) / myg.dy**2

        # Schlieren
        S.v()[:,:] = np.log(abs(discrete_Laplacian(rho)))
        S.d[S.d < -5.] = -6.

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


        fields = [rho, magvel, p, S]
        field_names = [r"$\rho$", r"$|u|$", "$p$", "$\ln(\mathcal{S})$"]
        colours = ['blue', 'red', 'black', 'green']
        colourmaps = [None, None, None, plt.get_cmap('Greys')]

        for n in range(4):
            ax = axes.flat[2*n]
            ax2 = axes.flat[2*n+1]

            v = fields[n]
            ycntr = np.round(0.5 * myg.qy).astype(int)
            img = ax.imshow(np.transpose(v.v()),
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax], vmin=vmins[n], vmax=vmaxes[n], cmap=colourmaps[n])
            plt2 = ax2.plot(myg.x, v.d[:,ycntr], c=colours[n])
            ax2.set_xlim([myg.xmin, myg.xmax])
            ax2.set_ylim([vmins[n], vmaxes[n]])

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

            plt.colorbar(img, ax=ax, orientation=orientation, shrink=0.75)


        plt.figtext(0.05,0.0125, "n: %4d,   t = %10.5f" % (self.n, self.cc_data.t))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.4, wspace=0.1)
        #plt.tight_layout()

        plt.draw()
