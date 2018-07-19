import burgers_viscid.burgers_fluxes as flx
import burgers_viscid.BC as BC
import burgers
import mesh.boundary as bnd
import multigrid.MG as MG
from simulation_null import grid_setup, bc_setup
import importlib
import particles.particles as particles


class Simulation(burgers.Simulation):

    def initialize(self):
        """
        Initialize the grid and variables for burgers and set the initial
        conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        my_data = self.data_class(my_grid)

        bnd.define_bc("constant", BC.user, is_solid=False)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)
        my_data.register_var("xvel", bc_xodd)
        my_data.register_var("yvel", bc_yodd)
        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            self.particles = particles.Particles(self.cc_data, bc, n_particles)

        self.ivars = burgers.Variables(my_data)

        self.cc_data.add_derived(burgers.derive_primitives)

        # now set the initial conditions for the problem
        problem = importlib.import_module(
            "burgers_viscid.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        if self.verbose > 0:
            print(my_data)

    def evolve(self):
        """
        Evolve the viscid burgers equation through one timestep.
        """
        myg = self.cc_data.grid
        myd = self.cc_data.data

        dtdx = self.dt / myg.dx
        dtdy = self.dt / myg.dy

        nu = self.rp.get_param("burgers.visc")

        flux_x, flux_y = flx.unsplit_fluxes(
            self.cc_data, self.rp, self.ivars, self.dt)

        # advective update term
        A = myg.scratch_array(nvar=self.ivars.nvar)

        for n in range(self.ivars.nvar):
            A.v(n=n)[:, :] = dtdx * (
                flux_x.ip(1, n=n) - flux_x.v(n=n)) - \
                dtdy * (flux_y.jp(1, n=n) - flux_y.v(n=n))

        # solve diffusion equation with the advective source

        # setup the MG object -- we want to solve a Helmholtz equation
        # equation of the form:
        # (alpha - beta L) u = f
        #
        # with alpha = 1
        #      beta  = (dt/2) nu
        #      f     = u + (dt/2) nu L u
        #
        # this is the form that arises with a Crank-Nicolson discretization
        # of the diffusion equation.
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               xl_BC_type=self.cc_data.BCs['xvel'].xlb,
                               xr_BC_type=self.cc_data.BCs['xvel'].xrb,
                               yl_BC_type=self.cc_data.BCs['xvel'].ylb,
                               yr_BC_type=self.cc_data.BCs['xvel'].yrb,
                               alpha=1.0, beta=0.5 * self.dt * nu,
                               verbose=0)

        u = self.cc_data.get_var("xvel")
        v = self.cc_data.get_var("yvel")

        # calculate the diffusive flux term
        visc_flux = flx.viscous_flux(u, v, myg, nu, self.ivars)

        for n in range(self.ivars.nvar):

            # form the RHS: f = u + (dt/2) nu L u  (where L is the Laplacian)
            f = mg.soln_grid.scratch_array()
            f.v()[:, :] = myd.v(n=n) + \
                0.5 * self.dt * visc_flux.v(n=n) - A.v(n=n)

            mg.init_RHS(f)

            # initial guess is zeros
            mg.init_zeros()

            # solve the MG problem for the updated u
            mg.solve(rtol=1.e-10)
            # mg.smooth(mg.nlevels-1,100)

            # update the solution
            myd.v(n=n)[:, :] = mg.get_solution().v()

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

    def write_extras(self, f):
        """
        Output simulation-specific data to the h5py file f
        """

        # make note of the custom BC
        gb = f.create_group("BC")

        # the value here is the value of "is_solid"
        gb.create_dataset("constant", data=False)
