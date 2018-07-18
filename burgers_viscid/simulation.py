import burgers_viscid.burgers_fluxes as flx
import burgers
import multigrid.MG as MG


class Simulation(burgers.Simulation):

    def evolve(self):
        """
        Evolve the linear burgers equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """
        myg = self.cc_data.grid

        dtdx = self.dt / myg.dx
        dtdy = self.dt / myg.dy

        nu = self.rp.get_param("burgers.visc")

        flux_x, flux_y = flx.unsplit_fluxes(
            self.cc_data, self.rp, self.ivars, self.dt)

        # advective update term
        A = myg.scratch_array(nvar=self.ivars.nvar)

        for n in range(self.ivars.nvar):
            A.v(n=n)[:, :] = -dtdx * (
                flux_x.v(n=n) - flux_x.ip(1, n=n)) - \
                dtdy * (flux_y.v(n=n) - flux_y.jp(1, n=n))

        # solve diffusion equation with the advective source
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               xl_BC_type=self.cc_data.BCs['xvel'].xlb,
                               xr_BC_type=self.cc_data.BCs['xvel'].xrb,
                               yl_BC_type=self.cc_data.BCs['xvel'].ylb,
                               yr_BC_type=self.cc_data.BCs['xvel'].yrb,
                               alpha=1.0, beta=0.5 * self.dt * nu,
                               verbose=0)

        q = self.cc_data.data

        for n in range(self.ivars.nvar):

            f = mg.soln_grid.scratch_array()
            f.v()[:, :] = q.v(n=n)  + \
                0.5 * self.dt * nu * (
                (q.ip(1, n=n) + q.ip(-1, n=n) -
                 2.0 * q.v(n=n)) / myg.dx**2 +
                (q.jp(1, n=n) + q.jp(-1, n=n) -
                 2.0 * q.v(n=n)) / myg.dy**2)- A.v(n=n)

            mg.init_RHS(f)

            # initial guess is zeros
            mg.init_zeros()

            # solve the MG problem for the updated phi
            mg.solve(rtol=1.e-10)
            #mg.smooth(mg.nlevels-1,100)

            # update the solution
            q.v(n=n)[:, :] = mg.get_solution().v()

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1
