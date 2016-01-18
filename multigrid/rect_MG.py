"""
This multigrid solver is build from multigrid/generalMG.py
and implements a rectangular solver.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import multigrid.edge_coeffs as ec
import multigrid.MG as MG
import multigrid.general_MG as general_MG
from copy import deepcopy

np.set_printoptions(precision=3, linewidth=128)


class RectMG2d(general_MG.GeneralMG2d):
    """
    this is a multigrid solver that supports our general elliptic
    equation.

    We need to accept a coefficient CellCenterData2d object with
    fields defined for alpha, beta, gamma_x, and gamma_y on the
    fine level.

    We then restrict this data through the MG hierarchy (and
    average beta to the edges).

    We need a new solve() function that implementes a more complex bottom
    solve than the current smoothing done on a 2x2 grid.
    """

    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                 yl_BC_type="dirichlet", yr_BC_type="dirichlet",
                 xl_BC=None, xr_BC=None,
                 yl_BC=None, yr_BC=None,
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0,
                 coeffs=None,
                 true_function=None, vis=0, vis_title=""):
        """
        here, coeffs is a CCData2d object
        """

        self.nx = nx
        self.ny = ny

        self.ng = ng

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.alpha = alpha
        self.beta = beta

        self.nsmooth = nsmooth
        self.nsmooth_bottom = nsmooth_bottom

        self.max_cycles = 100

        self.verbose = verbose

        # for visualization purposes, we can set a function name that
        # provides the true solution to our elliptic problem.
        if not true_function == None:
            self.true_function = true_function

        # a small number used in computing the error, so we don't divide by 0
        self.small = 1.e-16

        # keep track of whether we've initialized the RHS
        self.initialized_RHS = 0

        self.nlevels = int(math.log(min(self.nx, self.ny))/math.log(2.0))

        # a multigrid object will be a list of grids
        self.grids = []

        # create the grids.  Here, self.grids[0] will be the coarsest
        # grid and self.grids[nlevel-1] will be the finest grid
        # we store the solution, v, the rhs, f.

        # create the boundary condition object
        bc = patch.BCObject(xlb=xl_BC_type, xrb=xr_BC_type,
                            ylb=yl_BC_type, yrb=yr_BC_type)

        # we're going to assume that nx = a*2^n, ny = b*2^m, but that
        # nx = ny is not necessarily true
        if self.nx == 2**(self.nlevels-1):
            nx_t = 2
        else:
            nx_t = nx / (2**(self.nlevels-1))
        if self.ny == 2**(self.nlevels-1):
            ny_t = 2
        else:
            ny_t = ny / (2**(self.nlevels-1))

        for i in range(self.nlevels):

            # create the grid
            my_grid = patch.Grid2d(nx_t, ny_t, ng=self.ng,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

            # add a CellCenterData2d object for this level to our list
            self.grids.append(patch.CellCenterData2d(my_grid, dtype=np.float64))

            # create the phi BC object -- this only applies for the finest
            # level.  On the coarser levels, phi represents the residual,
            # which has homogeneous BCs
            bc_p = patch.BCObject(xlb=xl_BC_type, xrb=xr_BC_type,
                                  ylb=yl_BC_type, yrb=yr_BC_type,
                                  xl_func=xl_BC, xr_func=xr_BC,
                                  yl_func=yl_BC, yr_func=yr_BC, grid=my_grid)

            if i == self.nlevels-1:
                self.grids[i].register_var("v", bc_p)
            else:
                self.grids[i].register_var("v", bc)

            self.grids[i].register_var("f", bc)
            self.grids[i].register_var("r", bc)

            if not aux_field == None:
                for f, b in zip(aux_field, aux_bc):
                    self.grids[i].register_var(f, b)

            self.grids[i].create()

            if self.verbose: print(self.grids[i])

            nx_t = nx_t*2
            ny_t = ny_t*2

        # provide coordinate and indexing information for the solution mesh
        soln_grid = self.grids[self.nlevels-1].grid

        self.ilo = soln_grid.ilo
        self.ihi = soln_grid.ihi
        self.jlo = soln_grid.jlo
        self.jhi = soln_grid.jhi

        self.x  = soln_grid.x
        self.dx = soln_grid.dx
        self.x2d = soln_grid.x2d

        self.y  = soln_grid.y
        self.dy = soln_grid.dy
        self.y2d = soln_grid.y2d

        self.soln_grid = soln_grid

        # store the source norm
        self.source_norm = 0.0

        # after solving, keep track of the number of cycles taken, the
        # relative error from the previous cycle, and the residual error
        # (normalized to the source norm)
        self.num_cycles = 0
        self.residual_error = 1.e33
        self.relative_error = 1.e33

        # keep track of where we are in the V
        self.current_cycle = -1
        self.current_level = -1
        self.up_or_down = ""

        # for visualization -- what frame are we outputting?
        self.vis = vis
        self.vis_title = vis_title
        self.frame = 0

        # the coefficents come in a dictionary.  Set the coefficients
        # and restrict them down the hierarchy we only need to do this
        # once.  We need to hold the original coeffs in our grid so we
        # can do a ghost cell fill.
        for c in ["alpha", "beta", "gamma_x", "gamma_y"]:
            v = self.grids[self.nlevels-1].get_var(c)
            v.v()[:,:] = coeffs.get_var(c).v()

            self.grids[self.nlevels-1].fill_BC(c)

            n = self.nlevels-2
            while n >= 0:
                f_patch = self.grids[n+1]
                c_patch = self.grids[n]

                coeffs_c = c_patch.get_var(c)
                coeffs_c.v()[:,:] = f_patch.restrict(c).v()

                self.grids[n].fill_BC(c)
                n -= 1


        # put the beta coefficients on edges
        beta = self.grids[self.nlevels-1].get_var("beta")
        self.beta_edge.insert(0, ec.EdgeCoeffs(self.grids[self.nlevels-1].grid, beta))

        n = self.nlevels-2
        while n >= 0:
            self.beta_edge.insert(0, self.beta_edge[0].restrict())
            n -= 1


    def solve(self, rtol = 1.e-11, fortran=True):
        """
        The main driver for the multigrid solution of the Helmholtz
        equation.  This controls the V-cycles, smoothing at each
        step of the way and uses simple smoothing at the coarsest
        level to perform the bottom solve.

        Parameters
        ----------
        rtol : float
            The relative tolerance (residual norm / source norm) to
            solve to.  Note that if the source norm is 0 (e.g. the
            righthand side of our equation is 0), then we just use
            the norm of the residual.

        """

        # start by making sure that we've initialized the RHS
        if not self.initialized_RHS:
            msg.fail("ERROR: RHS not initialized")

        # for now, we will just do V-cycles, continuing until we
        # achieve the L2 norm of the relative solution difference is <
        # rtol
        if self.verbose:
            print("source norm = ", self.source_norm)

        old_solution = self.grids[self.nlevels-1].get_var("v").copy()

        converged = 0
        cycle = 1

        while not converged and cycle <= self.max_cycles:

            self.current_cycle = cycle

            # zero out the solution on all but the finest grid
            for level in range(self.nlevels-1):
                v = self.grids[level].zero("v")

            # descending part
            if self.verbose:
                print("<<< beginning V-cycle (cycle %d) >>>\n" % cycle)

            level = self.nlevels-1
            while level > 0:

                self.current_level = level
                self.up_or_down = "down"

                fP = self.grids[level]
                cP = self.grids[level-1]

                if self.verbose:
                    self._compute_residual(level)

                    print("  level: {}, grid: {} x {}".format(level, fP.grid.nx, fP.grid.ny))
                    print("  before G-S, residual L2: {}".format(fP.get_var("r").norm() ))

                # smooth on the current level
                self.smooth(level, self.nsmooth, fortran=fortran)


                # compute the residual
                self._compute_residual(level)

                if self.verbose:
                    print("  after G-S, residual L2: {}".format(fP.get_var("r").norm() ))


                # restrict the residual down to the RHS of the coarser level
                f_coarse = cP.get_var("f")
                f_coarse.v()[:,:] = fP.restrict("r").v()

                level -= 1


            # solve the discrete coarse problem.  We could use any
            # number of different matrix solvers here (like CG), but
            # since we are 2x2 by design at this point, we will just
            # smooth
            if self.verbose: print("  bottom solve:")

            self.current_level = 0
            bP = self.grids[0]

            if self.verbose:
                print("  level = {}, nx = {}, ny = {}\n".format(
                    level, bP.grid.nx, bP.grid.ny))

            # CHANGED: CG solver?
            self.cG()

            bP.fill_BC("v")


            # ascending part
            for level in range(1,self.nlevels):

                self.current_level = level
                self.up_or_down = "up"

                fP = self.grids[level]
                cP = self.grids[level-1]

                # prolong the error up from the coarse grid
                e = cP.prolong("v")

                # correct the solution on the current grid
                v = fP.get_var("v")
                v.v()[:,:] += e.v()

                fP.fill_BC("v")

                if self.verbose:
                    self._compute_residual(level)

                    print("  level = {}, nx = {}, ny = {}".format(
                        level, fP.grid.nx, fP.grid.ny))

                    print("  before G-S, residual L2: {}".format(fP.get_var("r").norm() ))

                # smooth
                self.smooth(level, self.nsmooth, fortran=fortran)

                if self.verbose:
                    self._compute_residual(level)

                    print("  after G-S, residual L2: {}".format(fP.get_var("r").norm() ))


            # compute the error with respect to the previous solution
            # this is for diagnostic purposes only -- it is not used to
            # determine convergence
            solnP = self.grids[self.nlevels-1]

            diff = (solnP.get_var("v").v() - old_solution.v())/ \
                (solnP.get_var("v").v() + self.small)

            relative_error = solnP.grid.norm(diff)

            old_solution = solnP.get_var("v").copy()

            # compute the residual error, relative to the source norm
            self._compute_residual(self.nlevels-1)
            r = fP.get_var("r")

            if self.source_norm != 0.0:
                residual_error = r.norm()/self.source_norm
            else:
                residual_error = r.norm()

            if residual_error < rtol:
                converged = 1
                self.num_cycles = cycle
                self.relative_error = relative_error
                self.residual_error = residual_error
                fP.fill_BC("v")

            if self.verbose:
                print("cycle {}: relative err = {}, residual err = {}\n".format(
                      cycle, relative_error, residual_error))

            cycle += 1

    def cG(self):
        """
        Implements conjugate gradient method for bottom solve.
        """

        bP = self.grids[0]

        self._compute_residual(0)

        r = bP.get_var("r")
        p = deepcopy(r)
        xfl = bP.get_var("v").v().flatten()
        myg = bP.grid
        rsold = np.inner(p.v().flatten(), p.v().flatten())

        def Ax(y):
            return self.alpha * y.v() - \
                self.beta * ((y.ip(-1) + y.ip(1) - 2*y.v())/myg.dx**2 +
                             (y.jp(-1) + y.jp(1) - 2*y.v())/myg.dy**2)
        for i in range(len(p.v().flatten())):
            Ap = Ax(p.v().flatten())
            a = rsold / np.inner(p.v().flatten(), Ap)
            xfl[:] += a * p.v().flatten()
            rfl = r.v().flatten() - a * Ap
            rsnew = np.inner(rfl, rfl)
            if np.sqrt(rsnew) < 1.e10:
                break
            p.v()[:,:] = np.reshape(rfl + (rsnew / rsold) * p.v().flatten(), (myg.nx, myg.ny))
            rsold = rsnew

        x = bP.get_var("v")
        x.v()[:,:] = np.reshape(xfl, (myg.nx, myg.ny))
