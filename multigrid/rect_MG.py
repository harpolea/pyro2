from __future__ import print_function

import numpy as np

import multigrid.edge_coeffs as ec
import multigrid.mg_utils_f as mg_f
import mesh.patch as patch
from multigrid.variable_coeff_MG import VarCoeffCCMG2d
import math
#from functools import partial
from util import msg
from functools import partial

np.set_printoptions(precision=3, linewidth=128)

class RectMG2d(VarCoeffCCMG2d):
    """
    this is a multigrid solver that supports rectangular grids
    """

    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                 yl_BC_type="dirichlet", yr_BC_type="dirichlet",
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0,
                 coeffs_x=None, coeffs_y=None, coeffs_bc=None,
                 true_function=None, vis=0, vis_title="", R=1.0, rp=None):

        # we'll keep a list of the coefficients averaged to the interfaces
        # on each level -- note: this will already be scaled by 1/dx**2
        self.edge_coeffs = []
        self.nx = nx
        self.ny = ny

        self.ng = 1#ng

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.alpha = 0.0
        self.beta = 0.0

        self.nsmooth = nsmooth
        self.nsmooth_bottom = nsmooth_bottom

        self.max_cycles = 100

        self.verbose = verbose

        if not true_function == None:
            self.true_function = true_function

        # a small number used in computing the error, so we don't divide by 0
        self.small = 1.e-16

        # keep track of whether we've initialized the RHS
        self.initialized_RHS = 0

        if self.nx < self.ny:
            self.nlevels = int(math.log(self.nx)/math.log(2.0))
        else:
            self.nlevels = int(math.log(self.ny)/math.log(2.0))

        #self.nlevels = min(int(math.log(self.nx)/math.log(2.0)), int(math.log(self.ny)/math.log(2.0)))

        nx_t = self.nx / (2**(self.nlevels-1))
        ny_t = self.ny / (2**(self.nlevels-1))

        # create the boundary condition object
        bc = patch.BCObject(xlb=xl_BC_type, xrb=xr_BC_type,
                            ylb=yl_BC_type, yrb=yr_BC_type)

        # redo some stuff
        self.grids = []

        for i in range(self.nlevels):

            # create the grid
            my_grid = patch.Grid2d(nx_t, ny_t, ng=self.ng,
                                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, R=R)

            # add a CellCenterData2d object for this level to our list
            self.grids.append(patch.CellCenterData2d(my_grid, dtype=np.float64))

            # create the phi BC object -- this only applies for the finest
            # level.  On the coarser levels, phi represents the residual,
            # which has homogeneous BCs
            bc_p = patch.BCObject(xlb=xl_BC_type, xrb=xr_BC_type,
                                  ylb=yl_BC_type, yrb=yr_BC_type, grid=my_grid)

            if i == self.nlevels-1:
                self.grids[i].register_var("v", bc_p)
            else:
                self.grids[i].register_var("v", bc)

            self.grids[i].register_var("f", bc)
            self.grids[i].register_var("r", bc)

            aux_field = ['coeffs_x', 'coeffs_y']
            aux_bc = [coeffs_bc, coeffs_bc]
            for f, b in zip(aux_field, aux_bc):
                self.grids[i].register_var(f, b)

            self.grids[i].create()
            # set up metric - use the fact that coeffs is a CCData2d object already which will have a grid
            self.grids[i].grid.initialise_metric(rp, coeffs_x.g.metric.alpha,
                coeffs_x.g.metric.beta,
                coeffs_x.g.metric.gamma, cartesian=False)

            if self.verbose:
                print(self.grids[i])

            nx_t *= 2
            ny_t *= 2

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
        self.dy = soln_grid.dy   # note, dy = dx is assumed
        self.y2d = soln_grid.y2d

        self.R  = soln_grid.R
        self.r2d = soln_grid.r2d

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

        # set the coefficients and restrict them down the hierarchy
        # we only need to do this once.  We need to hold the original
        # coeffs in our grid so we can do a ghost cell fill.
        c_x = self.grids[self.nlevels-1].get_var("coeffs_x")
        c_x.v()[:,:] = coeffs_x.v()[:,:]
        c_y = self.grids[self.nlevels-1].get_var("coeffs_y")
        c_y.v()[:,:] = coeffs_y.v()[:,:]

        self.grids[self.nlevels-1].fill_BC("coeffs_x")
        self.grids[self.nlevels-1].fill_BC("coeffs_y")

        # put the coefficients on edges
        self.edge_coeffs.insert(0, ec.EdgeCoeffs(self.grids[self.nlevels-1].grid, c_x, etay=c_y))

        #n = self.nlevels-2
        for n in range(self.nlevels-2, -1, -1): #while n >= 0:

            # create the edge coefficients on level n by restricting from the
            # finer grid
            f_patch = self.grids[n+1]
            c_patch = self.grids[n]

            coeffs_cx = c_patch.get_var("coeffs_x")
            coeffs_cy = c_patch.get_var("coeffs_y")
            coeffs_cx.v()[:,:] = f_patch.restrict("coeffs_x").v()
            coeffs_cy.v()[:,:] = f_patch.restrict("coeffs_y").v()

            self.grids[n].fill_BC("coeffs_x")
            self.grids[n].fill_BC("coeffs_y")

            # put the coefficients on edges
            self.edge_coeffs.insert(0, self.edge_coeffs[0].restrict())

            #n -= 1

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

        converged = False
        cycle = 1

        while not converged and cycle <= self.max_cycles:

            self.current_cycle = cycle

            # zero out the solution on all but the finest grid
            for level in range(self.nlevels-1):
                v = self.grids[level].zero("v")

            # descending part
            if self.verbose:
                print("<<< beginning V-cycle (cycle %d) >>>\n" % cycle)

            #level = self.nlevels-1
            for level in range(self.nlevels-1, 0, -1): #while for level > 0:

                self.current_level = level
                self.up_or_down = "down"

                fP = self.grids[level]
                cP = self.grids[level-1]

                if self.verbose:
                    self._compute_residual(level)

                    print("  level: {}, grid: {} x {}".format(level, fP.grid.nx, fP.grid.ny))
                    print("  before G-S, residual L2: {}".format(fP.get_var("r").norm() ))

                # smooth on the current level
                self.smooth(level, self.nsmooth)

                # compute the residual
                self._compute_residual(level)

                if self.verbose:
                    print("  after G-S, residual L2: {}".format(fP.get_var("r").norm() ))

                # restrict the residual down to the RHS of the coarser level
                f_coarse = cP.get_var("f")
                f_coarse.v()[:,:] = fP.restrict("r").v()

                #level -= 1


            # solve the discrete coarse problem.
            if self.verbose:
                print("  bottom solve:")

            self.current_level = 0
            bP = self.grids[0]

            if self.verbose:
                print("  level = {}, nx = {}, ny = {}\n".format(
                    level, bP.grid.nx, bP.grid.ny))

            if bP.grid.ny == bP.grid.nx and bP.grid.dx == bP.grid.dy: # square so can just use smoothing
                self.smooth(0, self.nsmooth_bottom, fortran=fortran)
            else:
                self.cG()
                # do we need this here??
                #self.smooth(0, self.nsmooth_bottom, fortran=fortran)

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
                self.smooth(level, self.nsmooth)

                if self.verbose:
                    self._compute_residual(level)

                    print("  after G-S, residual L2: {}".format(fP.get_var("r").norm() ))


            # compute the error with respect to the previous solution
            # this is for diagnostic purposes only -- it is not used to
            # determine convergence
            solnP = self.grids[self.nlevels-1]

            diff = solnP.grid.scratch_array()
            diff.v()[:,:] = (solnP.get_var("v").v() - old_solution.v())/ \
                (solnP.get_var("v").v() + self.small)

            relative_error = solnP.grid.norm(diff.d)

            old_solution = solnP.get_var("v").copy()

            # compute the residual error, relative to the source norm
            self._compute_residual(self.nlevels-1)
            r = fP.get_var("r")

            if self.source_norm != 0.0:
                residual_error = r.norm()/self.source_norm
            else:
                residual_error = r.norm()

            if residual_error < rtol:
                converged = True
                self.num_cycles = cycle
                self.relative_error = relative_error
                self.residual_error = residual_error
                fP.fill_BC("v")

            if self.verbose:
                print("cycle {}: relative err = {}, residual err = {}\n".format(
                      cycle, relative_error, residual_error))

            cycle += 1


    def L_eta_phi(self, myg, eta_x, eta_y, y):

        return (
        (# eta_{i+1/2,j} (phi_{i+1,j} - phi_{i,j})
        eta_x[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] *
        (y[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
         y[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]) -
        # eta_{i-1/2,j} (phi_{i,j} - phi_{i-1,j})
        eta_x[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] *
        (y[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] -
         y[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1]) ) +
        (# eta_{i,j+1/2} (phi_{i,j+1} - phi_{i,j})
        eta_y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] *
        (y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -  # y-diff
         y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])-
        # eta_{i,j-1/2} (phi_{i,j} - phi_{i,j-1})
        eta_y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] *
        (y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] -
         y[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])) )


    def cG(self, tol=1.e-10):
        """
        Implements conjugate gradient method for bottom solve.

        This is based off the method found here https://en.wikipedia.org/wiki/Conjugate_gradient_method
        """

        bP = self.grids[0]
        myg = bP.grid

        self._compute_residual(0)

        r = bP.get_var("r")
        p = myg.scratch_array()
        p.v()[:,:] = r.v()
        rfl = r.v().flatten()
        xfl = bP.get_var("v").v().flatten()

        rsold = np.inner(rfl, rfl)

        eta_x = self.edge_coeffs[0].x.d
        eta_y = self.edge_coeffs[0].y.d

        if 1 > 2:

            bcs = ['outflow', 'outflow','outflow', 'outflow']

            # note: this might be breaking as L eta phi operates on ghost
            # cells of p, which are *not* updated according to BCs at each
            # iteration

            for i in range(len(rfl)):
                Ap = self.L_eta_phi(myg, eta_x, eta_y, p.d).flatten()
                #print('Ap: {}'.format(Ap))
                denom = np.inner(p.v().flatten(), Ap)
                #print('denom: {}'.format(denom))
                if rsold == 0.0:
                    a = 0.0
                else:
                    a = rsold / denom
                #print('a: {}'.format(a))
                xfl += a * p.v().flatten()
                rfl -= a * Ap
                rsnew = np.inner(rfl, rfl)
                #print('rsold: {}'.format(rsnew))
                if np.sqrt(rsnew) < tol:
                    break
                p.v()[:,:] = np.reshape(rfl + (rsnew / rsold) * p.v().flatten(), (myg.nx, myg.ny))

                # FIXME: this bit screws things up
                #bP.fill_BC_given_data(xfl, bcs)
                rsold = rsnew

            x = bP.get_var("v")
            x.v()[:,:] = np.reshape(xfl, (myg.nx, myg.ny))
        else:
            x = bP.get_var("v")
            x.d[:,:] = mg_f.cg_f(myg.qx, myg.qy, myg.ng,
                                 np.asfortranarray(eta_x), np.asfortranarray(eta_y), np.asfortranarray(bP.get_var("v").d),
                                 np.asfortranarray(bP.get_var("r").d), tol)

            #print(x.d)

    #def get_solution_gradient(self, grid=None):
        """
        Return the gradient of the solution after doing the MG solve.  The
        x- and y-components are returned in separate arrays.

        If a grid object is passed in, then the gradient is computed on that
        grid.

        Returns
        -------
        out : ndarray, ndarray



        myg = self.soln_grid

        if grid is None:
            og = self.soln_grid
        else:
            og = grid
            assert og.dx == myg.dx and og.dy == myg.dy

        v = self.grids[self.nlevels-1].get_var("v")

        if isinstance(og.metric.gamma, partial):
            gamma = og.metric.gamma(og)
        else:
            gamma = og.metric.gamma

        gx = og.scratch_array()
        gy = og.scratch_array()

        # upstairs index components
        g_xx = 1. / gamma[og.ilo:og.ihi+1,og.jlo:og.jhi+1,0,0]
        g_yy = 1. / gamma[og.ilo:og.ihi+1,og.jlo:og.jhi+1,1,1]

        gx.v()[:,:] = 0.5 * g_xx * (v.ip(1) - v.ip(-1))/ (og.dx * myg.r2v)
        gy.v()[:,:] = 0.5 * g_yy * (v.jp(1) - v.jp(-1)) / og.dy

        return gx, gy"""
