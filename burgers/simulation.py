import importlib
import numpy as np
import matplotlib.pyplot as plt

import burgers.burgers_fluxes as flx
# import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
import util.plot_tools as plot_tools
import particles.particles as particles


class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """

    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.ixmom = myd.names.index("xvel")
        self.iymom = myd.names.index("yvel")

        # if there are any additional variable, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 2
        if self.naux > 0:
            self.ix = 2
        else:
            self.ix = -1

        # primitive variables
        self.nq = 2 + self.naux

        self.iu = 0
        self.iv = 1


def derive_primitives(myd, varnames):
    """
    Return the velocity
    """

    derived_vars = []

    if isinstance(varnames, str):
        wanted = [varnames]
    else:
        wanted = list(varnames)

    for var in wanted:

        if var == "velocity":
            u = myd.get_var("xvel")
            v = myd.get_var("yvel")
            derived_vars.append(u)
            derived_vars.append(v)

    if len(derived_vars) > 1:
        return derived_vars
    else:
        return derived_vars[0]


class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for burgers and set the initial
        conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        my_data = self.data_class(my_grid)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)
        my_data.register_var("xvel", bc_xodd)
        my_data.register_var("yvel", bc_yodd)
        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            self.particles = particles.Particles(self.cc_data, bc, n_particles)

        self.ivars = Variables(my_data)

        self.cc_data.add_derived(derive_primitives)

        # now set the initial conditions for the problem
        problem = importlib.import_module(
            "burgers.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        if self.verbose > 0:
            print(my_data)

    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        u = self.cc_data.get_var("xvel")
        v = self.cc_data.get_var("yvel")

        # the timestep is min(dx/|u|, dy/|v|)
        xtmp = self.cc_data.grid.dx / np.maximum(abs(u), self.SMALL)
        ytmp = self.cc_data.grid.dy / np.maximum(abs(v), self.SMALL)

        self.dt = cfl * float(min(xtmp.min(), ytmp.min()))

    def evolve(self):
        """
        Evolve the inviscid Burgers equation through one timestep.
        """

        dtdx = self.dt / self.cc_data.grid.dx
        dtdy = self.dt / self.cc_data.grid.dy

        flux_x, flux_y = flx.unsplit_fluxes(
            self.cc_data, self.rp, self.ivars, self.dt)

        """
        do the differencing for the fluxes now.  Here, we use slices so we
        avoid slow loops in python.  This is equivalent to:

        myPatch.data[i,j] = myPatch.data[i,j] + \
                               dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                               dtdy*(flux_y[i,j] - flux_y[i,j+1])
        """

        q = self.cc_data.data

        for n in range(self.ivars.nvar):
            q.v(n=n)[:, :] = q.v(n=n) + dtdx * (flux_x.v(n=n) - flux_x.ip(1, n=n)) + \
                dtdy * (flux_y.v(n=n) - flux_y.jp(1, n=n))

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

    def dovis(self):
        """
        Do runtime visualization.
        """
        plt.clf()

        plt.rc("font", size=10)

        u = self.cc_data.get_var("xvel")
        v = self.cc_data.get_var("yvel")

        myg = self.cc_data.grid

        fields = [u, v]
        field_names = [r"$u$", r"$v$"]

        _, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        for n, ax in enumerate(axes):
            v = fields[n]

            img = ax.imshow(np.transpose(v.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        if self.particles is not None:
            ax = axes[0]
            particle_positions = self.particles.get_positions()
            # dye particles
            colors = self.particles.get_init_positions()[:, 0]

            # plot particles
            ax.scatter(particle_positions[:, 0],
                       particle_positions[:, 1], s=5, c=colors, alpha=0.8, cmap="Greys")
            ax.set_xlim([myg.xmin, myg.xmax])
            ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, "t = {:10.5f}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
