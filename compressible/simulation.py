from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import compressible.BC as BC
from compressible.problems import *
import compressible.eos as eos
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
from compressible.unsplitFluxessL import *
from util import profile
import pylsmlib

"""
TODO:    Add in burning and laminar speed calculation.

CHANGED: Have implemented the null initialisation from the latest version of
         vanilla pyro.
"""


class Variables:
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, idens=-1, ixmom=-1, iymom=-1, iener=-1, iphi=-1):
        self.nvar = 5

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.idens = idens
        self.ixmom = ixmom
        self.iymom = iymom
        self.iener = iener
        self.iphi = iphi

        # primitive variables
        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.ip = 3
        self.iph = 4


class Simulation(NullSimulation):
    """
    def __init__(self, problem_name, rp, timers=None):
        #
        Initialize the Simulation object for compressible hydrodynamics.

        Parameters
        ----------
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in compressible/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        #

        NullSimulation.__init__(self, solver_name, problem_name, rp,
        timers=timers)
    """

    def initialize(self):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """

        # create the variables
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

        # level-set field
        my_data.register_var("phi", bc)

        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        my_data.create()

        self.cc_data = my_data

        self.vars = Variables(idens=my_data.vars.index("density"),
                              ixmom=my_data.vars.index("x-momentum"),
                              iymom=my_data.vars.index("y-momentum"),
                              iener=my_data.vars.index("energy"),
                              iphi=my_data.vars.index("phi"))

        # initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')

        if self.verbose > 0:
            print(my_data)

    def timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        # we need to compute the pressure
        u = xmom/dens
        v = ymom/dens

        e = (ener - 0.5*dens*(u*u + v*v))/dens

        gamma = self.rp.get_param("eos.gamma")

        p = eos.pres(gamma, dens, e)

        # compute the sound speed
        cs = np.sqrt(gamma*p/dens)

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx/(abs(u) + cs)
        ytmp = self.cc_data.grid.dy/(abs(v) + cs)

        dt = cfl*min(xtmp.min(), ytmp.min())

        return dt


#    def preevolve(self):
        """
        Do any necessary evolution before the main evolve loop.  This
        is not needed for compressible flow.
        """
#        pass

    def evolve(self, dt):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")
        phi = self.cc_data.get_var("phi")

        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid

        Flux_x, Flux_y = unsplitFluxes(self.cc_data, self.rp, self.vars,
                                       self.tc, dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # conservative update
        dtdx = dt/myg.dx
        dtdy = dt/myg.dy

        for n in range(self.vars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] += \
                dtdx*(Flux_x[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1, n] -
                      Flux_x[myg.ilo+1:myg.ihi+2, myg.jlo:myg.jhi+1, n]) + \
                dtdy*(Flux_y[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1, n] -
                      Flux_y[myg.ilo:myg.ihi+1, myg.jlo+1:myg.jhi+2, n])

        # gravitational source terms
        ymom += 0.5*dt*(dens + old_dens)*grav
        ener += 0.5*dt*(ymom + old_ymom)*grav

        # reinitialise
        phi[:, :] = pylsmlib.computeDistanceFunction(phi,
                                                     dx=self.cc_data.grid.dx,
                                                     order=2)

        tm_evolve.end()

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")
        phi = self.cc_data.get_var("phi")

        # get the velocities
        u = xmom/dens
        v = ymom/dens

        # get the pressure
        magvel = u**2 + v**2   # temporarily |U|^2
        rhoe = (ener - 0.5*dens*magvel)

        magvel = np.sqrt(magvel)

        e = rhoe/dens

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        p = eos.pres(gamma, dens, e)

        myg = self.cc_data.grid

        # figure out the geometry
        L_x = self.cc_data.grid.xmax - self.cc_data.grid.xmin
        L_y = self.cc_data.grid.ymax - self.cc_data.grid.ymin

        orientation = "vertical"
        shrink = 1.0

        sparseX = 0
        allYlabel = 1

        if (L_x > 2*L_y):

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

        elif (L_y > 2*L_x):

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

            onLeft = [0, 2]

        fields = [dens, magvel, phi, e]
        field_names = [r"$\rho$", r"U", r"$\phi$", "e"]

        for n in range(4):
            ax = axes.flat[n]

            v = fields[n]
            if n == 2:
                img = ax.contour(np.transpose(v[myg.ilo:myg.ihi+1,
                                                myg.jlo:myg.jhi+1]),
                                 origin="lower",
                                 extent=[myg.xmin, myg.xmax, myg.ymin,
                                         myg.ymax])
            else:
                img = ax.imshow(np.transpose(v[myg.ilo:myg.ihi+1,
                                               myg.jlo:myg.jhi+1]),
                                interpolation="nearest", origin="lower",
                                extent=[myg.xmin, myg.xmax, myg.ymin,
                                        myg.ymax])

            ax.set_xlabel("x")
            if n == 0:
                ax.set_ylabel("y")
            elif allYlabel:
                ax.set_ylabel("y")

            ax.set_title(field_names[n])

            if n not in onLeft:
                ax.yaxis.offsetText.set_visible(False)
                if n > 0:
                    ax.get_yaxis().set_visible(False)

            if sparseX:
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            plt.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)

        plt.figtext(0.05, 0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()
