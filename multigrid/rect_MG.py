"""
This multigrid solver is build from multigrid/generalMG.py
and implements a rectangular solver.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import multigrid.edge_coeffs_sph as ec
import multigrid.mg_utils_f as mg_f
#import multigrid.MG as MG
import multigrid.variable_coeff_MG as var_MG
#from copy import deepcopy
import math
import mesh.patch as patch
import mesh.patch_sph as patch_sph
from lm_gr.simulation import Basestate
from util import msg
#import mesh.metric as metric

np.set_printoptions(precision=3, linewidth=128)


class RectMG2d(var_MG.VarCoeffCCMG2d):
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
                 coeffs=None, coeffs_bc=None,
                 true_function=None, vis=0, vis_title="", R=0.0, cc=1.0, grav=0.0, rp=None):
        """
        here, coeffs is a CCData2d object
        """

        self.nx = nx
        self.ny = ny

        self.ng = 1

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.alpha = 0.0
        self.beta = 0.0

        self.rp = rp

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
        bc = patch_sph.BCObject_Sph(xlb=xl_BC_type, xrb=xr_BC_type,
                            ylb=yl_BC_type, yrb=yr_BC_type)

        # we're going to assume that nx = a*2^n, ny = b*2^m, but that
        # nx = ny is not necessarily true
        if self.nx == 2**self.nlevels:
            nx_t = 2
        else:
            nx_t = nx / (2**(self.nlevels-1))
        if self.ny == 2**self.nlevels:
            ny_t = 2
        else:
            ny_t = ny / (2**(self.nlevels-1))

        for i in range(self.nlevels):

            # create the grid
            my_grid = patch_sph.Grid2d_Sph(nx_t, ny_t, ng=self.ng,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, R=R)

            # set up metric - use the fact that coeffs is a CCData2d object already which will have a grid
            my_grid.initialise_metric(self.rp,
                coeffs.g.metric.alpha,
                coeffs.g.metric.beta,
                coeffs.g.metric.gamma,
                cartesian=coeffs.g.metric.cartesian)

            # add a CellCenterData2d object for this level to our list
            self.grids.append(patch_sph.CellCenterData2d_Sph(my_grid, dtype=np.float64))

            # create the phi BC object -- this only applies for the finest
            # level.  On the coarser levels, phi represents the residual,
            # which has homogeneous BCs
            bc_p = patch_sph.BCObject_Sph(xlb=xl_BC_type, xrb=xr_BC_type,
                                  ylb=yl_BC_type, yrb=yr_BC_type,
                                  xl_func=xl_BC, xr_func=xr_BC,
                                  yl_func=yl_BC, yr_func=yr_BC, grid=my_grid)

            if i == self.nlevels-1:
                self.grids[i].register_var("v", bc_p)
            else:
                self.grids[i].register_var("v", bc)

            self.grids[i].register_var("f", bc)
            self.grids[i].register_var("r", bc)
            aux_field = ["coeffs"]
            aux_bc = [coeffs_bc]

            for f, b in zip(aux_field, aux_bc):
                self.grids[i].register_var(f, b)

            self.grids[i].create()

            if self.verbose:
                print(self.grids[i])

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

        # set the coefficients and restrict them down the hierarchy
        # we only need to do this once.  We need to hold the original
        # coeffs in our grid so we can do a ghost cell fill.
        c = self.grids[self.nlevels-1].get_var("coeffs")
        c.v()[:,:] = coeffs.v()[:,:]

        self.grids[self.nlevels-1].fill_BC("coeffs")

        self.edge_coeffs = []

        # put the coefficients on edges
        self.edge_coeffs.insert(0, ec.EdgeCoeffsSpherical(self.grids[self.nlevels-1].grid, c))

        n = self.nlevels-2
        while n >= 0:

            # create the edge coefficients on level n by restricting from the
            # finer grid
            f_patch = self.grids[n+1]
            c_patch = self.grids[n]

            coeffs_c = c_patch.get_var("coeffs")
            coeffs_c.v()[:,:] = f_patch.restrict("coeffs").v()

            self.grids[n].fill_BC("coeffs")

            # put the coefficients on edges
            self.edge_coeffs.insert(0, self.edge_coeffs[0].restrict())

            n -= 1


    def smooth(self, level, nsmooth, fortran=True):
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

        """
        #FIXME: get rid
        #super(RectMG2d, self).smooth(level, nsmooth)
        #return

        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")

        myg = self.grids[level].grid

        eta_x = self.edge_coeffs[level].x.d
        eta_y = self.edge_coeffs[level].y.d

        if fortran:
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
                          nsmooth, np.asfortranarray(v.d), np.asfortranarray(f.d), bcints, np.asfortranarray(eta_x), np.asfortranarray(eta_y),
                          np.asfortranarray(myg.r2d), np.asfortranarray(myg.x2d), myg.dx, myg.dy)

            v.d[:,:] = (patch.ArrayIndexer(d=_v, grid=myg)).d

        else:
            self.grids[level].fill_BC("v")

            # print( "min/max c: {}, {}".format(np.min(c), np.max(c)))
            # print( "min/max eta_x: {}, {}".format(np.min(eta_x), np.max(eta_x)))
            # print( "min/max eta_y: {}, {}".format(np.min(eta_y), np.max(eta_y)))


            # do red-black G-S
            for i in range(nsmooth):

                # do the red black updating in four decoupled groups
                #
                #
                #        |       |       |
                #      --+-------+-------+--
                #        |       |       |
                #        |   4   |   3   |
                #        |       |       |
                #      --+-------+-------+--
                #        |       |       |
                #   jlo  |   1   |   2   |
                #        |       |       |
                #      --+-------+-------+--
                #        |  ilo  |       |
                #
                # groups 1 and 3 are done together, then we need to
                # fill ghost cells, and then groups 2 and 4

                for n, (ix, iy) in enumerate([(0,0), (1,1), (1,0), (0,1)]):

                    denom = (
                        (eta_x[myg.ilo+1+ix:myg.ihi+2+ix:2,
                               myg.jlo+iy  :myg.jhi+1+iy:2] +
                        #
                        eta_x[myg.ilo+ix  :myg.ihi+1+ix:2,
                              myg.jlo+iy  :myg.jhi+1+iy:2]) /
                        (myg.r2d[myg.ilo+ix  :myg.ihi+1+ix:2,
                                 myg.jlo+iy  :myg.jhi+1+iy:2] *
                         np.sin(myg.x2d[myg.ilo+ix  :myg.ihi+1+ix:2,
                                myg.jlo+iy  :myg.jhi+1+iy:2]) * myg.dx) +
                        #
                        (eta_y[myg.ilo+ix  :myg.ihi+1+ix:2,
                              myg.jlo+1+iy:myg.jhi+2+iy:2] +
                        #
                        eta_y[myg.ilo+ix  :myg.ihi+1+ix:2,
                              myg.jlo+iy  :myg.jhi+1+iy:2]) /
                        (myg.r2d[myg.ilo+ix  :myg.ihi+1+ix:2,
                                 myg.jlo+iy  :myg.jhi+1+iy:2]**2 * myg.dy))

                    v.d[myg.ilo+ix:myg.ihi+1+ix:2,myg.jlo+iy:myg.jhi+1+iy:2] = (
                        -f.d[myg.ilo+ix:myg.ihi+1+ix:2,
                             myg.jlo+iy:myg.jhi+1+iy:2] +
                        # eta_{i+1/2,j} phi_{i+1,j}
                        (eta_x[myg.ilo+1+ix:myg.ihi+2+ix:2,
                               myg.jlo+iy  :myg.jhi+1+iy:2] *
                        v.d[myg.ilo+1+ix:myg.ihi+2+ix:2,
                            myg.jlo+iy  :myg.jhi+1+iy:2] +
                        # eta_{i-1/2,j} phi_{i-1,j}
                        eta_x[myg.ilo+ix:myg.ihi+1+ix:2,
                              myg.jlo+iy:myg.jhi+1+iy:2]*
                        v.d[myg.ilo-1+ix:myg.ihi+ix  :2,
                            myg.jlo+iy  :myg.jhi+1+iy:2]) /
                        (myg.r2d[myg.ilo+ix  :myg.ihi+1+ix:2,
                                 myg.jlo+iy  :myg.jhi+1+iy:2] *
                         np.sin(myg.x2d[myg.ilo+ix  :myg.ihi+1+ix:2,
                                myg.jlo+iy  :myg.jhi+1+iy:2]) * myg.dx) +
                        # eta_{i,j+1/2} phi_{i,j+1}
                        (eta_y[myg.ilo+ix:myg.ihi+1+ix:2,
                               myg.jlo+1+iy:myg.jhi+2+iy:2]*
                        v.d[myg.ilo+ix  :myg.ihi+1+ix:2,
                            myg.jlo+1+iy:myg.jhi+2+iy:2] +
                        # eta_{i,j-1/2} phi_{i,j-1}
                        eta_y[myg.ilo+ix:myg.ihi+1+ix:2,
                              myg.jlo+iy:myg.jhi+1+iy:2]*
                        v.d[myg.ilo+ix  :myg.ihi+1+ix:2,
                            myg.jlo-1+iy:myg.jhi+iy  :2]) /
                        (myg.r2d[myg.ilo+ix  :myg.ihi+1+ix:2,
                                 myg.jlo+iy  :myg.jhi+1+iy:2]**2 *
                         myg.dy)) / denom

                    if n == 1 or n == 3:
                        self.grids[level].fill_BC("v")

            if self.vis == 1:
                plt.clf()

                plt.subplot(221)
                self._draw_solution()

                plt.subplot(222)
                self._draw_V()

                plt.subplot(223)
                self._draw_main_solution()

                plt.subplot(224)
                self._draw_main_error()

                plt.suptitle(self.vis_title, fontsize=18)

                plt.draw()
                plt.savefig("mg_%4.4d.png" % (self.frame))
                self.frame += 1


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
        #FIXME: get rid
        #super(RectMG2d, self).solve()
        #return

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
                self.smooth(level, self.nsmooth)

                # compute the residual
                self._compute_residual(level)

                if self.verbose:
                    print("  after G-S, residual L2: {}".format(fP.get_var("r").norm() ))

                # restrict the residual down to the RHS of the coarser level
                f_coarse = cP.get_var("f")
                f_coarse.v()[:,:] = fP.restrict("r").v()

                level -= 1


            # solve the discrete coarse problem.
            if self.verbose:
                print("  bottom solve:")

            self.current_level = 0
            bP = self.grids[0]

            if self.verbose:
                print("  level = {}, nx = {}, ny = {}\n".format(
                    level, bP.grid.nx, bP.grid.ny))

            if bP.grid.ny == bP.grid.nx: # square so can just use smoothing
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

    def cG(self, tol=1.e-10):
        """
        Implements conjugate gradient method for bottom solve.

        This is based off the method found here https://en.wikipedia.org/wiki/Conjugate_gradient_method
        """

        bP = self.grids[0]
        myg = bP.grid
        dx = myg.dx
        dy = myg.dy

        self._compute_residual(0)

        r = bP.get_var("r")
        p = myg.scratch_array()
        p.v()[:,:] = r.v()[:,:]
        rfl = r.v().flatten()
        xfl = bP.get_var("v").v().flatten()

        rsold = np.inner(rfl, rfl)

        eta_x = self.edge_coeffs[0].x.d
        eta_y = self.edge_coeffs[0].y.d

        xs = myg.scratch_array()
        xs.d[:,:] = myg.x2d

        rs = myg.scratch_array()
        rs.d[:,:] = myg.r2d

        bcs = ['outflow', 'outflow','outflow', 'outflow']

        # note: this might be breaking as L eta phi operates on ghost
        # cells of p, which are *not* updated according to BCs at each
        # iteration
        def L_eta_phi(y):
            return (
            (# eta_{i+1/2,j} (phi_{i+1,j} - phi_{i,j})
            eta_x[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] *
            (y[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             y[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]) -
            # eta_{i-1/2,j} (phi_{i,j} - phi_{i-1,j})
            eta_x[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] *
            (y[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] -
             y[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1]) ) / (dx * myg.r2v * np.sin(myg.x2v)) +
            (# eta_{i,j+1/2} (phi_{i,j+1} - phi_{i,j})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] *
            (y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -  # y-diff
             y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])-
            # eta_{i,j-1/2} (phi_{i,j} - phi_{i,j-1})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] *
            (y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] -
             y[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])) / (dy * myg.r2v**2) )


        for i in range(len(rfl)):
            Ap = L_eta_phi(p.d).flatten()
            #print('Ap: {}'.format(Ap))
            if rsold == 0.0:
                a = 0.0
            else:
                a = rsold / np.inner(p.v().flatten(), Ap)
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


    def get_solution_gradient_sph(self, grid=None):
        """
        Return the gradient of the solution after doing the MG solve.  The
        x- and y-components are returned in separate arrays.

        If a grid object is passed in, then the gradient is computed on
        that grid.

        grad f = df/dr e_r + (1/r) df/dtheta e_theta

        Returns
        -------
        out : ndarray, ndarray

        """

        myg = self.soln_grid

        if grid is None:
            og = self.soln_grid
        else:
            og = grid
            assert og.dx == myg.dx and og.dy == myg.dy

        v = self.grids[self.nlevels-1].get_var("v")

        gx = og.scratch_array()
        gy = og.scratch_array()

        alphasq = Basestate(og.ny, ng=og.ng)
        alphasq.d[:] = myg.metric.alpha(og).d**2

        # FIXME: do we need the r**2 here??
        # upstairs index components
        g_xx = alphasq.d2df(og.qx) / og.r2d**2
        g_yy = alphasq.d2df(og.qx)

        gx.v()[:,:] = 0.5 * g_xx[og.ilo:og.ihi+1,og.jlo:og.jhi+1] * (v.ip(1) - v.ip(-1)) / (og.dx * og.r2v)
        gy.v()[:,:] = 0.5 * g_yy[og.ilo:og.ihi+1,og.jlo:og.jhi+1] *(v.jp(1) - v.jp(-1)) / og.dy

        return gx, gy


    def _compute_residual(self, level):
        """ compute the residual and store it in the r variable"""
        #FIXME: get rid
        #super(RectMG2d, self)._compute_residual(level)
        #return

        v = self.grids[level].get_var("v").d
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")

        myg = self.grids[level].grid

        eta_x = self.edge_coeffs[level].x.d
        eta_y = self.edge_coeffs[level].y.d

        dx = myg.dx
        dy = myg.dy

        # compute the residual
        # r = f - L_eta phi
        L_eta_phi = (
            (# eta_{i+1/2,j} (phi_{i+1,j} - phi_{i,j})
            eta_x[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] *
            (v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             v[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]) -
            # eta_{i-1/2,j} (phi_{i,j} - phi_{i-1,j})
            eta_x[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] *
            (v[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] -
             v[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1]) ) / (dx * myg.r2v * np.sin(myg.x2v)) +
            (# eta_{i,j+1/2} (phi_{i,j+1} - phi_{i,j})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] *
            (v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -  # y-diff
             v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])-
            # eta_{i,j-1/2} (phi_{i,j} - phi_{i,j-1})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] *
            (v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] -
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])) / (dy * myg.r2v**2) )

        #print('L_eta_phi: {}'.format(L_eta_phi))

        r.v()[:,:] = f.v() - L_eta_phi
