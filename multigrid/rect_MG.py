from __future__ import print_function

import numpy as np

import multigrid.edge_coeffs as ec
import multigrid.mg_utils_f as mg_f
import mesh.patch as patch
from multigrid.variable_coeff_MG import VarCoeffCCMG2d
import math
from functools import partial
from util import msg

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
                 coeffs=None, coeffs_bc=None,
                 true_function=None, vis=0, vis_title="", R=1.0, rp=None):

        # we'll keep a list of the coefficients averaged to the interfaces
        # on each level -- note: this will already be scaled by 1/dx**2
        self.edge_coeffs = []

        # initialize the MG object with the auxillary "coeffs" field
        super(RectMG2d, self).__init__(nx, ny,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                   xl_BC_type=xl_BC_type, xr_BC_type=xr_BC_type,
                                   yl_BC_type=yl_BC_type, yr_BC_type=yr_BC_type,
                                   nsmooth=nsmooth, nsmooth_bottom=nsmooth_bottom,
                                   verbose=verbose,
                                   coeffs=coeffs, coeffs_bc=coeffs_bc,
                                   true_function=true_function,
                                   vis=vis, vis_title=vis_title)

        self.nlevels = max(int(math.log(self.nx)/math.log(2.0)), int(math.log(self.ny)/math.log(2.0)))

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

            aux_field = ['coeffs']
            aux_bc = [coeffs_bc]
            for f, b in zip(aux_field, aux_bc):
                self.grids[i].register_var(f, b)

            self.grids[i].create()
            # set up metric - use the fact that coeffs is a CCData2d object already which will have a grid
            self.grids[i].grid.initialise_metric(rp, coeffs.g.metric.alpha,
                coeffs.g.metric.beta,
                coeffs.g.metric.gamma, cartesian=False)

            if self.verbose:
                print(self.grids[i])

            nx_t *= 2
            ny_t *= 2

        # provide coordinate and indexing information for the solution mesh
        soln_grid = self.grids[self.nlevels-1].grid

        self.R  = soln_grid.R
        self.r2d = soln_grid.r2d

        self.soln_grid = soln_grid


        # set the coefficients and restrict them down the hierarchy
        # we only need to do this once.  We need to hold the original
        # coeffs in our grid so we can do a ghost cell fill.
        c = self.grids[self.nlevels-1].get_var("coeffs")
        c.v()[:,:] = coeffs.v()[:,:]

        self.grids[self.nlevels-1].fill_BC("coeffs")

        # put the coefficients on edges
        self.edge_coeffs.insert(0, ec.EdgeCoeffs(self.grids[self.nlevels-1].grid, c))

        #n = self.nlevels-2
        for n in range(self.nlevels-2, -1, -1): #while n >= 0:

            # create the edge coefficients on level n by restricting from the
            # finer grid
            f_patch = self.grids[n+1]
            c_patch = self.grids[n]

            coeffs_c = c_patch.get_var("coeffs")
            coeffs_c.v()[:,:] = f_patch.restrict("coeffs").v()

            self.grids[n].fill_BC("coeffs")

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

    #def smooth(self, level, nsmooth, fortran=True):
        """
        Use red-black Gauss-Seidel iterations to smooth the solution
        at a given level.  This is used at each stage of the V-cycle
        (up and down) in the MG solution, but it can also be called
        directly to solve the elliptic problem (although it will take
        many more iterations).

        Parameters
        ----------
        level : int
            The level in the MG hierarchy to smooth the solution
        nsmooth : int
            The number of r-b Gauss-Seidel smoothing iterations to perform



        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")

        myg = self.grids[level].grid

        eta_x = self.edge_coeffs[level].x.d
        eta_y = self.edge_coeffs[level].y.d

        # convert bcs into fotran-compatible version
        bcs = [self.grids[level].BCs["v"].xlb,
               self.grids[level].BCs["v"].xrb,
               self.grids[level].BCs["v"].ylb,
               self.grids[level].BCs["v"].yrb]
        bcints = 3 * np.ones(4, dtype=np.int)

        for i in range(4):
            if bcs[i] in ["outflow", "neumann"]:
                bcints[i] = 0
            elif bcs[i] == "reflect-even":
                bcints[i] = 1
            elif bcs[i] in ["reflect-odd", "dirichlet"]:
                bcints[i] = 2
            elif bcs[i] == "periodic":
                bcints[i] = 3

        _v = mg_f.smooth_sph_f(myg.qx, myg.qy, myg.ng,
                      nsmooth, np.asfortranarray(v.d),
                      np.asfortranarray(f.d), bcints,
                      np.asfortranarray(eta_x), np.asfortranarray(eta_y),
                      np.asfortranarray(myg.r2d),
                      np.asfortranarray(myg.x2d), myg.dx, myg.dy)

        v.d[:,:] = (patch.ArrayIndexer(d=_v, grid=myg)).d"""


    #def _compute_residual(self, level):
        """ compute the residual and store it in the r variable"""

        """v = self.grids[level].get_var("v").d
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")

        myg = self.grids[level].grid

        eta_x = self.edge_coeffs[level].x.d
        eta_y = self.edge_coeffs[level].y.d

        # compute the residual
        # r = f - L_eta phi

        L_eta_phi = self.L_eta_phi(myg, eta_x, eta_y, v)

        r.v()[:,:] = f.v() - L_eta_phi"""


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
