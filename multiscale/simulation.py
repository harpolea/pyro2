from __future__ import print_function

import importlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

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
        swe_data.set_aux("g", self.rp.get_param("multiscale.grav"))
        swe_data.set_aux("rhobar", self.rp.get_param("swe.rhobar"))
        comp_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        comp_data.set_aux("grav", self.rp.get_param("multiscale.grav"))
        comp_data.set_aux("z", self.rp.get_param("compressible.z"))

        boundary_loc = self.rp.get_param("multiscale.boundary_loc")
        boundary_loc = int(self.rp.get_param("mesh.nx") * boundary_loc)

        swe_data.set_aux("boundary_loc", boundary_loc)
        comp_data.set_aux("boundary_loc", boundary_loc)

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
        problem.init_data(self.swe_data, self.comp_data,
                          self.aux_data, self.rp)

        if self.verbose > 0:
            print(swe_data)
            print(comp_data)

    def swe_to_comp(self):
        """
        Calculate ghost data in swe then convert to compressible
        """

        swg = self.swe_data.grid
        cog = self.comp_data.grid

        swd = self.swe_data.data
        cod = self.comp_data.data

        cvars = self.ivars_comp
        svars = self.ivars_swe

        sq = swe.cons_to_prim(swd, self.rp, svars, swg)
        cq = comp.cons_to_prim(cod, self.rp, cvars, cog)

        g = self.swe_data.get_aux("g")
        rhobar = self.swe_data.get_aux("rhobar")
        z = self.comp_data.get_aux("z")

        # x-dir
        i = self.swe_data.get_aux("boundary_loc")

        for j in range(swg.ny):

            qc = cq[i + 1, j, :]
            qs = sq[i, j, :]

            C_SWE = (qc[cvars.irho] * (qc[cvars.iu] - qs[svars.iu]) -
                     (qc[cvars.ip] - rhobar * g * (qs[svars.ih] - z))) / (rhobar + qc[cvars.irho])

            q_ghost = qs[:]
            q_ghost[svars.ih] -= C_SWE
            q_ghost[svars.iu] += C_SWE * qs[svars.iu]

            swd[i + 1, j, :] = swe.prim_to_cons_vec(q_ghost, self.rp, svars)

        # we need to make sure enough of the ghost cells have been updated, so we'll just copy
        # the column that has been found
        for g in range(1, swg.ng):
            swd[i + 1 + g, :, :] = swd[i + 1, :, :]

    def comp_to_swe(self):
        """
        Calculate ghost data in compressible then convert to swe
        """

        swg = self.swe_data.grid
        cog = self.comp_data.grid

        swd = self.swe_data.data
        cod = self.comp_data.data

        cs = self.comp_data.get_var("soundspeed")

        cvars = self.ivars_comp
        svars = self.ivars_swe

        sq = swe.cons_to_prim(swd, self.rp, svars, swg)
        cq = comp.cons_to_prim(cod, self.rp, cvars, cog)

        g = self.swe_data.get_aux("g")
        rhobar = self.swe_data.get_aux("rhobar")
        z = self.comp_data.get_aux("z")

        # x-dir
        i = self.swe_data.get_aux("boundary_loc")

        for j in range(swg.ny):

            qc = cq[i + 1, j, :]
            qs = sq[i, j, :]

            C_SWE = (qc[cvars.irho] * (qc[cvars.iu] - qs[svars.iu]) -
                     (qc[cvars.ip] - rhobar * g * (qs[svars.ih] - z))) / (rhobar + qc[cvars.irho])

            C_Euler = qc[cvars.irho] / cs[i + 1, j]**2 * \
                ((qc[cvars.iu] - qs[cvars.iu]) - C_SWE)

            q_ghost = qc[:]
            q_ghost[cvars.ip] -= C_Euler * cs[i + 1, j]**2
            q_ghost[cvars.irho] -= C_Euler
            q_ghost[cvars.iu] -= C_Euler * \
                cs[i + 1, j] / qc[cvars.irho]

            cod[i, j, :] = comp.prim_to_cons_vec(q_ghost, self.rp, cvars)

        # we need to make sure enough of the ghost cells have been updated, so we'll just copy
        # the column that has been found
        for g in range(1, cog.ng):
            cod[i - g, :, :] = cod[i, :, :]

    def fix_ghosts(self):

        # first average swe in vertical direction
        vertical_average = np.average(self.swe_data.data, 1)
        self.swe_data.data[:, :, :] = vertical_average[:, np.newaxis, :]

        self.swe_to_comp()
        self.comp_to_swe()

        # average again
        vertical_average = np.average(self.swe_data.data, 1)
        self.swe_data.data[:, :, :] = vertical_average[:, np.newaxis, :]

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
        u, cs = self.swe_data.get_var(["velocity", "soundspeed"])
        U, v, CS = self.comp_data.get_var(["velocity", "soundspeed"])
        u = np.maximum(u, U)
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

        self.fix_ghosts()

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
        svars = swe.Variables(self.swe_data)
        cvars = comp.Variables(self.comp_data)

        qs = swe.cons_to_prim(self.swe_data.data, self.rp,
                             svars, self.swe_data.grid)

        qc = comp.cons_to_prim(self.comp_data.data, self.rp,
                          cvars, self.comp_data.grid)

        i = self.swe_data.get_aux("boundary_loc")

        h = qs[:, :, svars.ih]
        u = qs[:, :, svars.iu]
        v = qs[:, :, svars.iu]
        fuel = qs[:, :, svars.ix]

        dens = qc[:,:,cvars.idens]
        uc = qc[:,:,cvars.iu]
        vc = qc[:,:,cvars.iv]
        p = qc[:,:,cvars.ip]

        # magvel = np.sqrt(u**2 + v**2)

        myg = self.comp_data.grid

        # vort = myg.scratch_array()
        #
        # dv = 0.5 * (v.ip(1) - v.ip(-1)) / myg.dx
        # du = 0.5 * (u.jp(1) - u.jp(-1)) / myg.dy

        # vort.v()[:, :] = dv - du

        swe_fields = [h, u, v, fuel]
        swe_field_names = [r"$h$", r"$u$", r"$v$", r"$X$"]

        comp_fields = [dens, uc, vc, p]
        comp_field_names = [r"$\rho$", r"$u$", r"$v$", r"$p$"]

        f = plt.figure(1)
        f.set_size_inches(12,8)
        axes = AxesGrid(f, 111,
                        nrows_ncols=(4, 2),
                        share_all=True,
                        cbar_mode="each",
                        cbar_location="top",
                        cbar_pad="10%",
                        cbar_size="25%",
                        axes_pad=(0, 0.85),
                        add_all=True, label_mode="L")
        cbar_title = True

        # _, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        for n in range(4):
            ax_swe = axes[2*n]
            ax_comp = axes[2*n+1]
            swe_v = swe_fields[n]
            comp_v = comp_fields[n]

            img = ax_swe.imshow(np.transpose(swe_v[:i, :].v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax/2, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax_swe.set_xlabel("x")
            ax_swe.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[2*n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(swe_field_names[n], pad=25)
            else:
                ax_swe.set_title(swe_field_names[n])

            img2 = ax_comp.imshow(np.transpose(comp_v[i:, :].v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax/2, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax_comp.set_xlabel("x")
            ax_comp.set_ylabel("y")

            # needed for PDF rendering
            cb_c = axes.cbar_axes[2*n+1].colorbar(img2)
            cb_c.solids.set_rasterized(True)
            cb_c.solids.set_edgecolor("face")

            if cbar_title:
                cb_c.ax.set_title(comp_field_names[n], pad=25)
            else:
                ax_comp.set_title(comp_field_names[n])

            smin, smax = cb.get_clim()
            cmin, cmax = cb_c.get_clim()

            if n == 2 or n == 3:
                cb.set_clim(min(smin, cmin), max(smax, cmax))
                cb_c.set_clim(min(smin, cmin), max(smax, cmax))


        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
