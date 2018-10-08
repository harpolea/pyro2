from __future__ import print_function

import importlib

import numpy as np
import matplotlib.pyplot as plt

import mesh.boundary as bnd
from simulation_null import NullSimulation, grid_setup, bc_setup
import util.plot_tools as plot_tools
import particles.particles as particles
import multiscale.swe as swe
import multiscale.compressible as comp
import multiscale.unsplit_fluxes as flx


class Simulation(NullSimulation):
    """The main simulation class for the corner transport upwind
    swe hydrodynamics solver

    """

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for swe flow and set
        the initial conditions for the chosen problem.
        """
        my_grid = grid_setup(self.rp, ng=ng)
        swe_data = self.data_class(my_grid)
        comp_data = self.data_class(my_grid)
        cc_data = self.data_class(my_grid)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid(bc)

        swe_data.register_var("height", bc)
        swe_data.register_var("x-momentum", bc_xodd)
        swe_data.register_var("y-momentum", bc_yodd)
        swe_data.register_var("fuel", bc)

        comp_data.register_var("density", bc)
        comp_data.register_var("energy", bc)
        comp_data.register_var("x-momentum", bc_xodd)
        comp_data.register_var("y-momentum", bc_yodd)
        comp_data.register_var("fuel", bc)

        # any extras?
        if extra_vars is not None:
            for v in extra_vars:
                swe_data.register_var(v, bc)
                comp_data.register_var(v, bc)

        # store the gravitational acceration g as an auxillary quantity
        # so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        swe_data.set_aux("g", self.rp.get_param("swe.grav"))
        comp_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        comp_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        swe_data.create()
        comp_data.create()
        cc_data.create()

        self.swe_data = swe_data
        self.comp_data = comp_data
        self.cc_data = cc_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param(
                "particles.particle_generator")
            self.particles = particles.Particles(
                self.cc_data, bc, n_particles, particle_generator)

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = self.data_class(my_grid)
        aux_data.register_var("ymom_src", bc_yodd)
        aux_data.register_var("E_src", bc)
        aux_data.register_var("multiscale_mask", bc)
        aux_data.create()
        self.aux_data = aux_data

        self.ivars_swe = swe.Variables(swe_data)
        self.ivars_comp = comp.Variables(comp_data)

        # derived variables
        self.swe_data.add_derived(swe.derive_primitives)
        self.comp_data.add_derived(comp.derive_primitives)

        # initial conditions for the problem
        problem = importlib.import_module("{}.problems.{}".format(
            self.solver_name, self.problem_name))
        problem.init_data(self.swe_data, self.comp_data, self.rp)

        if self.verbose > 0:
            print(swe_data)
            print(comp_data)

    def method_compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        u, v, cs = self.swe_data.get_var(["velocity", "soundspeed"])
        U, V, CS = self.comp_data.get_var(["velocity", "soundspeed"])

        print(f"swe cs = {cs}, comp cs = {CS}")

        u = np.maximum(u, U)
        v = np.maximum(v, V)
        cs = np.maximum(cs, CS)

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx / (abs(u) + cs)
        ytmp = self.cc_data.grid.dy / (abs(v) + cs)

        self.dt = cfl * float(min(xtmp.min(), ytmp.min()))

    def evolve(self):
        """
        Evolve the equations of swe hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        for lib, data in zip([swe, comp], [self.swe_data, self.comp_data]):

            myg = data.grid

            Flux_x, Flux_y = flx.unsplit_fluxes(lib, data, self.aux_data, self.rp,
                                                self.solid, self.tc, self.dt)

            # conservative update
            dtdx = self.dt / myg.dx
            dtdy = self.dt / myg.dy

            nvar = len(data.names)

            for n in range(nvar):
                var = data.get_var_by_index(n)

                var.v()[:, :] += \
                    dtdx * (Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                    dtdy * (Flux_y.v(n=n) - Flux_y.jp(1, n=n))

            # increment the time
            data.t += self.dt

            # fill boundary conditions
            data.fill_BC_all()

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        self.n += 1
        self.cc_data.t += self.dt

        tm_evolve.end()

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        # we do this even though ivars is in self, so this works when
        # we are plotting from a file
        ivars = swe.Variables(self.swe_data)

        q = swe.cons_to_prim(self.swe_data.data, self.rp,
                             ivars, self.swe_data.grid)

        h = q[:, :, ivars.ih]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        fuel = q[:, :, ivars.ix]

        magvel = np.sqrt(u**2 + v**2)

        myg = self.swe_data.grid

        vort = myg.scratch_array()

        dv = 0.5 * (v.ip(1) - v.ip(-1)) / myg.dx
        du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy

        vort.v()[:, :] = dv - du

        fields = [h, magvel, fuel, vort]
        field_names = [r"$h$", r"$|U|$", r"$X$", r"$\nabla\times U$"]

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

        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
