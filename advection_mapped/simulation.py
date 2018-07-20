import importlib
import numpy as np
import matplotlib.pyplot as plt

import advection_mapped.advective_fluxes as flx
import mapped.mapped_grid as mapped
from simulation_null import NullSimulation, grid_setup, bc_setup
import util.plot_tools as plot_tools


class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """

        cart_grid = grid_setup(self.rp, ng=4)
        grid_type = self.rp.get_param("grid.map_type")

        if grid_type == "curvilinear":
            hxmin = self.rp.get_param("grid.rmin")
            hymin = self.rp.get_param("grid.thmin")
            hxmax = self.rp.get_param("grid.rmax")
            my_grid = mapped.Curvilinear(
                cart_grid, hxmin=hxmin, hxmax=hxmax, hymin=hymin, hymax=0.5 * np.pi)
        elif grid_type == "rectilinear":
            # hxmin = self.rp.get_param("grid.rxmin")
            # hymin = self.rp.get_param("grid.rymin")
            # hxmax = self.rp.get_param("grid.rxmax")
            # hxmax = self.rp.get_param("grid.rymax")
            my_grid = mapped.Rectilinear(
                cart_grid, hxmin=0, hxmax=2, hymin=0, hymax=1)
        else:
            my_grid = cart_grid

        # create the variables
        my_data = mapped.StructuredData2d(my_grid)
        bc = bc_setup(self.rp)[0]
        my_data.register_var("density", bc)
        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem
        problem = importlib.import_module(
            "advection_mapped.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp)

    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        u = self.rp.get_param("advection_mapped.u")
        v = self.rp.get_param("advection_mapped.v")

        # the timestep is min(dx/|u|, dy/|v|)
        xtmp = self.cc_data.grid.dhx.min() / max(abs(u), self.SMALL)
        ytmp = self.cc_data.grid.dhy.min() / max(abs(v), self.SMALL)

        self.dt = cfl * min(xtmp, ytmp)

    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """

        myg = self.cc_data.grid

        flux_x, flux_y = flx.unsplit_fluxes(
            self.cc_data, self.rp, self.dt, "density")

        """
        do the differencing for the fluxes now.  Here, we use slices so we
        avoid slow loops in python.  This is equivalent to:

        myPatch.data[i,j] = myPatch.data[i,j] + \
                               dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                               dtdy*(flux_y[i,j] - flux_y[i,j+1])
        """

        dens = self.cc_data.get_var("density")

        kappa = myg.scratch_array()

        kappa[:, :] = myg.cell_areas / (myg.cart.dx * myg.cart.dy)

        # print(f'kappa = {kappa}')

        dens.v()[:, :] = dens.v() - \
            self.dt / (kappa.v() * myg.cart.dx) * (flux_x.ip(1) - flux_x.v()) - \
            self.dt / (kappa.v() * myg.cart.dy) * (flux_y.jp(1) - flux_y.v())

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

    def dovis(self):
        """
        Do runtime visualization.
        """
        plt.clf()

        dens = self.cc_data.get_var("density")

        myg = self.cc_data.grid

        _, axes, cbar_title = plot_tools.setup_axes(
            myg, 2, share_all=False, force_cols=True)

        # plot density
        ax = axes[0]
        img = ax.imshow(np.transpose(dens.v()),
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                        cmap=self.cm)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim([myg.ymin, myg.ymax])

        # needed for PDF rendering
        cb = axes.cbar_axes[0].colorbar(img)
        cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")

        # plot density
        ax = axes[1]
        img = ax.imshow(np.transpose(dens.v()),
                        interpolation="nearest", origin="lower",
                        extent=[myg.hxmin, myg.hxmax, myg.hymin, myg.hymax],
                        cmap=self.cm)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim([myg.hymin, myg.hymax])

        # needed for PDF rendering
        cb = axes.cbar_axes[1].colorbar(img)
        cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")

        plt.title("density")

        plt.figtext(0.05, 0.0125, "t = {:10.5f}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
