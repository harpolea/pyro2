"""
The multigrid module provides a framework for solving elliptic
problems.  A multigrid object is just a list of grids, from the finest
mesh down (by factors of two) to a single interior zone (each grid has
the same number of guardcells).

The main multigrid class is setup to solve a constant-coefficient
Helmholtz equation:

(alpha - beta L) phi = f

where L is the Laplacian and alpha and beta are constants.  If alpha =
0 and beta = -1, then this is the Poisson equation.

We support homogeneous Dirichlet or Neumann BCs, or on periodic domain.

The general usage is as follows:

> a = multigrid.CellCenterMG2d(nx, ny, verbose=1, alpha=alpha, beta=beta)

this creates the multigrid object a, with a finest grid of nx by ny
zones and the default boundary condition types.  alpha and beta are
the coefficients of the Helmholtz equation.  Setting verbose = 1
causing debugging information to be output, so you can see the
residual errors in each of the V-cycles.

> a.init_zeros()

this initializes the solution vector with zeros (this is not necessary
if you just created the multigrid object, but it can be used to reset
the solution between runs on the same object).

> a.init_RHS(zeros((nx, ny), numpy.float64))

this initializes the RHS on the finest grid to 0 (Laplace's equation).
Any RHS can be set by passing through an array of (nx, ny) values here.

Then to solve, you just do:

> a.solve(rtol = 1.e-10)

where rtol is the desired tolerance (residual norm / source norm)

to access the final solution, use the getSolution method

v = a.get_solution()

For convenience, the grid information on the solution level is available as
attributes to the class,

a.ilo, a.ihi, a.jlo, a.jhi are the indices bounding the interior
of the solution array (i.e. excluding the guardcells).

a.x and a.y are the coordinate arrays
a.dx and a.dy are the grid spacings

"""

from __future__ import print_function

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import mesh.patch as patch
from util import msg

def _error(myg, r):
    """
    L2 norm of elements in r, multiplied by dx*dy to
    normalize
    """

    return np.sqrt(myg.dx * myg.dy * \
        np.sum((r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2).flat))


class CellCenterMG2dRect:
    """
    The main multigrid class for cell-centered data.

    No longer require that nx = ny be a power of 2 and dx = dy (unlike
    CellCenterMG2d).
    """

    def __init__(self, nx, ny, ng=1,
                 xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                 yl_BC_type="dirichlet", yr_BC_type="dirichlet",
                 alpha=0.0, beta=-1.0,
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0,
                 aux_field=None, aux_bc=None,
                 true_function=None, vis=0, vis_title=""):
        """
        Create the CellCenterMG2dRect object.  Note that this requires a
        grid to be a power of 2 in size.

        Parameters
        ----------
        nx : int
            number of cells in x-direction
        ny : int
            number of cells in y-direction.
        xmin : float, optional
            minimum physical coordinate in x-direction
        xmax : float, optional
            maximum physical coordinate in x-direction
        ymin : float, optional
            minimum physical coordinate in y-direction
        ymax : float, optional
            maximum physical coordinate in y-direction
        xl_BC_type : {'neumann', 'dirichlet'}, optional
            boundary condition to enforce on lower x face
        xr_BC_type : {'neumann', 'dirichlet'}, optional
            boundary condition to enforce on upper x face
        yl_BC_type : {'neumann', 'dirichlet'}, optional
            boundary condition to enforce on lower y face
        yr_BC_type : {'neumann', 'dirichlet'}, optional
            boundary condition to enforce on upper y face
        alpha : float, optional
            coefficient in Helmholtz equation (alpha - beta L) phi = f
        beta : float, optional
            coefficient in Helmholtz equation (alpha - beta L) phi = f
        nsmooth : int, optional
            number of smoothing iterations to be done at each intermediate
            level in the V-cycle (up and down)
        nsmooth_bottom : int, optional
            number of smoothing iterations to be done during the bottom
            solve
        verbose : int, optional
            increase verbosity during the solve (for verbose=1)
        aux_field : str, optional
            an extra fields to define and carry at each level.
            Useful for subclassing.
        aux_bc : BCObject, optional
            the boundary conditions corresponding to the aux field
        true_function : function, optional
            a function (of x,y) that provides the exact solution to
            the elliptic problem we are solving.  This is used only
            for visualization purposes
        vis : int, optional
            output a detailed visualization of every smoothing step
            all throughout the V-cycle (if vis=1)
        vis_title : string, optional
            a descriptive title to write on the visualization plots

        Returns
        -------
        out: CellCenterMG2d object

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

        # assume self.nx, self.ny are equal to 2^p, 2^q, such that
        # min(self.nx, self.ny) = 2^(nlevels-1).
        # This defines nlevels such that we end exactly on a 2xm grid
        self.nlevels = int(math.log(min(self.nx, self.ny))/math.log(2.0))

        # FIXME: do some kind of validation to check nx,ny are both 2^n.

        # a multigrid object will be a list of grids
        self.grids = []

        # create the grids.  Here, self.grids[0] will be the coarsest
        # grid and self.grids[nlevel-1] will be the finest grid
        # we store the solution, v, the rhs, f.
        i = 0
        #nx_t = ny_t = 2
        # the smaller of nx, ny will have n_t starting at 2, the other shall be
        # 2 * some integer
        nx_t = self.nx / (2**(self.nlevels-1))
        ny_t = self.ny / (2**(self.nlevels-1))
        #print('nx: ', self.nx, 'ny: ', self.ny, 'nlevels:', self.nlevels, 'nx_t: ', nx_t, 'ny_t: ', ny_t)

        if self.verbose:
            print("alpha = ", self.alpha)
            print("beta  = ", self.beta)

        while i < self.nlevels:

            # create the grid
            my_grid = patch.Grid2d(nx_t, ny_t, ng=self.ng,
                                   xmin=xmin, xmax=xmax,
                                   ymin=ymin, ymax=ymax)

            # add a CellCenterData2d object for this level to our list
            self.grids.append(patch.CellCenterData2d(my_grid, dtype=np.float64))

            # create the boundary condition object
            bc = patch.BCObject(xlb=xl_BC_type, xrb=xr_BC_type,
                                ylb=yl_BC_type, yrb=yr_BC_type)

            self.grids[i].register_var("v", bc)
            self.grids[i].register_var("f", bc)
            self.grids[i].register_var("r", bc)

            if not aux_field == None:
                self.grids[i].register_var(aux_field, aux_bc)


            self.grids[i].create()

            if self.verbose:
                print(self.grids[i])

            # increase grid size and counter
            nx_t = nx_t*2
            ny_t = ny_t*2

            i += 1


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

    # these draw functions are for visualization purposes and are
    # not ordinarily used, except for plotting the progression of the
    # solution within the V
    def _draw_V(self):
        """ draw the V-cycle on our optional visualization """
        xdown = np.linspace(0.0, 0.5, self.nlevels)
        xup = np.linspace(0.5, 1.0, self.nlevels)

        ydown = np.linspace(1.0, 0.0, self.nlevels)
        yup = np.linspace(0.0, 1.0, self.nlevels)

        plt.plot(xdown, ydown, lw=2, color="k")
        plt.plot(xup, yup, lw=2, color="k")

        plt.scatter(xdown, ydown, marker="o", color="k", s=40)
        plt.scatter(xup, yup, marker="o", color="k", s=40)

        if self.up_or_down == "down":
            plt.scatter(xdown[self.nlevels-self.current_level-1], ydown[self.nlevels-self.current_level-1],
                          marker="o", color="r", zorder=100, s=38)

        else:
            plt.scatter(xup[self.current_level], yup[self.current_level],
                          marker="o", color="r", zorder=100, s=38)

        plt.text(0.7, 0.1, "V-cycle %d" % (self.current_cycle))
        plt.axis("off")


    def _draw_solution(self):
        """ plot the current solution on our optional visualization """
        myg = self.grids[self.current_level].grid

        v = self.grids[self.current_level].get_var("v")

        plt.imshow(np.transpose(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                     interpolation="nearest", origin="lower",
                     extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        #plt.xlabel("x")
        plt.ylabel("y")


        if self.current_level == self.nlevels-1:
            plt.title(r"solving $L\phi = f$")
        else:
            plt.title(r"solving $Le = r$")

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb = plt.colorbar(format=formatter, shrink=0.5)

        cb.ax.yaxis.offsetText.set_fontsize("small")
        cl = plt.getp(cb.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize="small")


    def _draw_main_solution(self):
        """
        plot the solution at the finest level on our optional
        visualization
        """
        myg = self.grids[self.nlevels-1].grid

        v = self.grids[self.nlevels-1].get_var("v")

        plt.imshow(np.transpose(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                     interpolation="nearest", origin="lower",
                     extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        plt.xlabel("x")
        plt.ylabel("y")


        plt.title(r"current fine grid solution")

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb = plt.colorbar(format=formatter, shrink=0.5)

        cb.ax.yaxis.offsetText.set_fontsize("small")
        cl = plt.getp(cb.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize="small")


    def _draw_main_error(self):
        """
        plot the error with respect to the true solution on our optional
        visualization
        """
        myg = self.grids[self.nlevels-1].grid

        v = self.grids[self.nlevels-1].get_var("v")

        e = v - self.true_function(myg.x2d, myg.y2d)

        plt.imshow(np.transpose(e[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                     interpolation="nearest", origin="lower",
                     extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        plt.xlabel("x")
        plt.ylabel("y")


        plt.title(r"current fine grid error")

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb = plt.colorbar(format=formatter, shrink=0.5)

        cb.ax.yaxis.offsetText.set_fontsize("small")
        cl = plt.getp(cb.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize="small")


    def get_solution(self, grid=None):
        """
        Return the solution after doing the MG solve

        If a grid object is passed in, then the gradient is computed on that
        grid.

        Returns
        -------
        out : ndarray

        """

        if grid is None:
            og = self.soln_grid
        else:
            og = grid

        v = self.grids[self.nlevels-1].get_var("v")

        myg = self.soln_grid

        if grid is None:
            return v.copy()
        else:
            sol = og.scratch_array()
            sol[og.ilo-1:og.ihi+2,og.jlo-1:og.jhi+2] = \
                v[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
            return sol

    def get_solution_gradient(self, grid=None):
        """
        Return the gradient of the solution after doing the MG solve.  The
        x- and y-components are returned in separate arrays.

        If a grid object is passed in, then the gradient is computed on that
        grid.

        Returns
        -------
        out : ndarray, ndarray

        """

        if grid is None:
            og = self.soln_grid
        else:
            og = grid

        v = self.grids[self.nlevels-1].get_var("v")

        gx = og.scratch_array()
        gy = og.scratch_array()

        myg = self.soln_grid

        gx[og.ilo:og.ihi+1,og.jlo:og.jhi+1] = \
            0.5*(v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 v[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx

        gy[og.ilo:og.ihi+1,og.jlo:og.jhi+1] = \
            0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        return gx, gy


    def get_solution_object(self):
        """
        Return the full solution data object at the finest resolution
        after doing the MG solve

        Returns
        -------
        out : CellCenterData2d object

        """
        return self.grids[self.nlevels-1]


    def init_solution(self, data):
        """
        Initialize the solution to the elliptic problem by passing in
        a value for all defined zones

        Parameters
        ----------
        data : ndarray
            An array (of the same size as the finest MG level) with the
            values to initialize the solution to the elliptic problem.

        """
        v = self.grids[self.nlevels-1].get_var("v")
        v[:,:] = data.copy()


    def init_zeros(self):
        """
        Set the initial solution to zero
        """
        v = self.grids[self.nlevels-1].get_var("v")
        v[:,:] = 0.0


    def init_RHS(self, data):
        """
        Initialize the right hand side, f, of the Helmholtz equation
        (alpha - beta L) phi = f

        Parameters
        ----------
        data : ndarray
            An array (of the same size as the finest MG level) with the
            values to initialize the solution to the elliptic problem.

        """

        f = self.grids[self.nlevels-1].get_var("f")
        f[:,:] = data.copy()

        # store the source norm
        self.source_norm = _error(self.grids[self.nlevels-1].grid, f)

        if self.verbose:
            print("Source norm = ", self.source_norm)

        # note: if we wanted to do inhomogeneous Dirichlet BCs, we
        # would modify the source term, f, here to include a boundary
        # charge

        self.initialized_RHS = 1


    def _compute_residual(self, level):
        """ compute the residual and store it in the r variable"""

        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")

        myg = self.grids[level].grid

        # compute the residual
        # r = f - alpha phi + beta L phi
        r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - \
            self.alpha*v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + \
            self.beta*(
            (v[myg.ilo-1:myg.ihi  ,myg.jlo  :myg.jhi+1] +
             v[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1] -
             2.0*v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/(myg.dx*myg.dx) +
            (v[myg.ilo  :myg.ihi+1,myg.jlo-1:myg.jhi  ] +
             v[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             2.0*v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/(myg.dy*myg.dy) )


    def smooth(self, level, nsmooth):
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
        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")

        myg = self.grids[level].grid

        self.grids[level].fill_BC("v")

        xcoeff = self.beta/myg.dx**2
        ycoeff = self.beta/myg.dy**2

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

                v[myg.ilo+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi+1:2] = \
                   (f[myg.ilo+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi+1:2] +
                    xcoeff*(v[myg.ilo+1+ix:myg.ihi+2:2,
                              myg.jlo+iy  :myg.jhi+1:2] +
                            v[myg.ilo-1+ix:myg.ihi  :2,
                              myg.jlo+iy  :myg.jhi+1:2]) +
                    ycoeff*(v[myg.ilo+ix  :myg.ihi+1:2,
                              myg.jlo+1+iy:myg.jhi+2:2] +
                            v[myg.ilo+ix  :myg.ihi+1:2,
                              myg.jlo-1+iy:myg.jhi  :2]))/ \
                   (self.alpha + 2.0*xcoeff + 2.0*ycoeff)

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


    def solve(self, rtol = 1.e-11):
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
            level = 0
            while level < self.nlevels-1:
                v = self.grids[level].zero("v")
                level += 1

            # descending part
            if self.verbose:
                print("<<< beginning V-cycle (cycle %d) >>>\n" % cycle)

            level = self.nlevels-1
            while level > 0:

                self.current_level = level
                self.up_or_down = "down"

                fP = self.grids[level]
                cP = self.grids[level-1]

                # access to the residual
                r = fP.get_var("r")

                if self.verbose:
                    self._compute_residual(level)

                    print("  level = %d, nx = %d, ny = %d" %  \
                        (level, fP.grid.nx, fP.grid.ny))

                    print("  before G-S, residual L2 norm = %g" % \
                          (_error(fP.grid, r) ))

                # smooth on the current level
                self.smooth(level, self.nsmooth)


                # compute the residual
                self._compute_residual(level)

                if self.verbose:
                    print("  after G-S, residual L2 norm = %g\n" % \
                          (_error(fP.grid, r) ))


                # restrict the residual down to the RHS of the coarser level
                f_coarse = cP.get_var("f")
                f_coarse[:,:] = fP.restrict("r")

                level -= 1


            # solve the discrete coarse problem.  We could use any
            # number of different matrix solvers here (like CG), but
            # since we are 2x2 by design at this point, we will just
            # smooth
            if self.verbose:
                print("  bottom solve:")

            self.current_level = 0

            bP = self.grids[0]

            if self.verbose:
                print("  level = %d, nx = %d, ny = %d\n" %  \
                    (level, bP.grid.nx, bP.grid.ny))

            self.smooth(0, self.nsmooth_bottom)

            bP.fill_BC("v")


            # ascending part
            level = 1
            while level < self.nlevels:

                self.current_level = level
                self.up_or_down = "up"

                fP = self.grids[level]
                cP = self.grids[level-1]

                # prolong the error up from the coarse grid
                e = cP.prolong("v")

                # correct the solution on the current grid
                v = fP.get_var("v")
                v += e

                fP.fill_BC("v")

                if self.verbose:
                    self._compute_residual(level)
                    r = fP.get_var("r")

                    print("  level = %d, nx = %d, ny = %d" % \
                        (level, fP.grid.nx, fP.grid.ny))

                    print("  before G-S, residual L2 norm = %g" % \
                          (_error(fP.grid, r) ))

                # smooth
                self.smooth(level, self.nsmooth)

                if self.verbose:
                    self._compute_residual(level)

                    print("  after G-S, residual L2 norm = %g\n" % \
                          (_error(fP.grid, r) ))

                level += 1

            # compute the error with respect to the previous solution
            # this is for diagnostic purposes only -- it is not used to
            # determine convergence
            solnP = self.grids[self.nlevels-1]

            diff = (solnP.get_var("v") - old_solution)/ \
                (solnP.get_var("v") + self.small)

            relative_error = _error(solnP.grid, diff)

            old_solution = solnP.get_var("v").copy()

            # compute the residual error, relative to the source norm
            self._compute_residual(self.nlevels-1)
            r = fP.get_var("r")

            if self.source_norm != 0.0:
                residual_error = _error(fP.grid, r)/self.source_norm
            else:
                residual_error = _error(fP.grid, r)


            if residual_error < rtol:
                converged = 1
                self.num_cycles = cycle
                self.relative_error = relative_error
                self.residual_error = residual_error
                fP.fill_BC("v")

            if self.verbose:
                print("cycle %d: relative err = %g, residual err = %g\n" % \
                      (cycle, relative_error, residual_error))

            cycle += 1

        if not converged:
            print('not converged, residual error = ', residual_error)
