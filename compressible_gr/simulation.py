from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import compressible.BC as BC
from compressible.problems import *
import compressible.eos as eos
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
from compressible.unsplitFluxes import *
from util import profile


class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, iD=-1, iSx=-1, iSy=-1, itau=-1):
        self.nvar = 4

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.iD = iD
        self.iSx = iSx
        self.iSy = iSy
        self.itau = itau

        # primitive variables
        self.irho = iD
        self.iu = iSx
        self.iv = iSy
        self.ih = itau
        #self.ip = 4


class Simulation(NullSimulation):

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
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("x-momentum", bc_xodd)
        my_data.register_var("y-momentum", bc_yodd)


        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        my_data.create()

        self.cc_data = my_data

        self.vars = Variables(iD = my_data.vars.index("density"),
                              iSx = my_data.vars.index("x-momentum"),
                              iSy = my_data.vars.index("y-momentum"),
                              itau = my_data.vars.index("energy"))


        # initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')

        if self.verbose > 0: print(my_data)


    def compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        D = self.cc_data.get_var("density")
        Sx = self.cc_data.get_var("x-momentum")
        Sy = self.cc_data.get_var("y-momentum")
        tau = self.cc_data.get_var("energy")

        # we need to compute the pressure
        u = Sx/D
        v = Sy/D

        e = (tau - 0.5*D*(u*u + v*v))/D

        gamma = self.rp.get_param("eos.gamma")

        p = eos.pres(gamma, D, e)

        # compute the sounds speed
        cs = np.sqrt(gamma*p/D)


        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx/(abs(u) + cs)
        ytmp = self.cc_data.grid.dy/(abs(v) + cs)

        self.dt = cfl*min(xtmp.min(), ytmp.min())


    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        D = self.cc_data.get_var("density")
        Sy = self.cc_data.get_var("y-momentum")
        tau = self.cc_data.get_var("energy")

        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid

        Flux_x, Flux_y = unsplitFluxes(self.cc_data, self.rp, self.vars, self.tc, self.dt)

        old_D = D.copy()
        old_Sy = Sy.copy()

        # conservative update
        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy

        for n in range(self.vars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var.v()[:,:] += \
                dtdx*(Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy*(Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # gravitational source terms
        Sy.d[:,:] += 0.5*self.dt*(D.d[:,:] + old_D.d[:,:])*grav
        tau.d[:,:] += 0.5*self.dt*(Sy.d[:,:] + old_Sy.d[:,:])*grav

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()


    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        D = self.cc_data.get_var("density")
        Sx = self.cc_data.get_var("x-momentum")
        Sy = self.cc_data.get_var("y-momentum")
        tau = self.cc_data.get_var("energy")

        # get the velocities
        u = Sx/D
        v = Sy/D

        # get the pressure
        magvel = u**2 + v**2   # temporarily |U|^2
        rhoe = (tau - 0.5*D*magvel)

        magvel = np.sqrt(magvel)

        e = rhoe/D

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        p = eos.pres(gamma, D, e)

        myg = self.cc_data.grid


        # figure out the geometry
        L_x = self.cc_data.grid.xmax - self.cc_data.grid.xmin
        L_y = self.cc_data.grid.ymax - self.cc_data.grid.ymin

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
            fig, axes = plt.subplots(nrows=4, ncols=1, num=1)
            orientation = "horizontal"
            if (L_x > 4*L_y):
                shrink = 0.75

            onLeft = list(range(self.vars.nvar))


        elif L_y > 2*L_x:

            # we want 4 columns:  rho  |U|  p  e
            fig, axes = plt.subplots(nrows=1, ncols=4, num=1)
            if (L_y >= 3*L_x):
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


        fields = [D, magvel, p, e]
        field_names = [r"$\rho$", r"U", "p", "e"]

        for n in range(4):
            ax = axes.flat[n]

            v = fields[n]
            img = ax.imshow(np.transpose(v.v()),
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

            ax.set_xlabel("x")
            if n == 0:
                ax.set_ylabel("y")
            elif allYlabel:
                ax.set_ylabel("y")

            ax.set_title(field_names[n])

            if not n in onLeft:
                ax.yaxis.offsetText.set_visible(False)
                if n > 0: ax.get_yaxis().set_visible(False)

            if sparseX:
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            plt.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()
