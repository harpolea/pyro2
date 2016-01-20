"""
This multigrid solver is build from multigrid/MG.py and implements
a variable coefficient solver for an equation of the form:

div { eta grad phi } = f

where eta is defined on the same grid as phi.

A cell-centered discretization is used throughout.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import multigrid.MG as MG
import multigrid.edge_coeffs as ec
import multigrid.mg_utils_f as mg_f
import mesh.patch as patch

np.set_printoptions(precision=3, linewidth=128)


class VarCoeffCCMG2d(MG.CellCenterMG2d):
    """
    this is a multigrid solver that supports variable coefficients

    we need to accept a coefficient array, coeffs, defined at each
    level.  We can do this at the fine level and restrict it
    down the MG grids once.

    we need a new compute_residual() and smooth() function, that
    understands coeffs.
    """

    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                 yl_BC_type="dirichlet", yr_BC_type="dirichlet",
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0,
                 coeffs=None, coeffs_bc=None,
                 true_function=None, vis=0, vis_title=""):

        # we'll keep a list of the coefficients averaged to the interfaces
        # on each level -- note: this will already be scaled by 1/dx**2
        self.edge_coeffs = []

        # initialize the MG object with the auxillary "coeffs" field
        MG.CellCenterMG2d.__init__(self, nx, ny, ng=1,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                   xl_BC_type=xl_BC_type, xr_BC_type=xr_BC_type,
                                   yl_BC_type=yl_BC_type, yr_BC_type=yr_BC_type,
                                   alpha=0.0, beta=0.0,
                                   nsmooth=nsmooth, nsmooth_bottom=nsmooth_bottom,
                                   verbose=verbose,
                                   aux_field=["coeffs"], aux_bc=[coeffs_bc],
                                   true_function=true_function, vis=vis,
                                   vis_title=vis_title)


        # set the coefficients and restrict them down the hierarchy
        # we only need to do this once.  We need to hold the original
        # coeffs in our grid so we can do a ghost cell fill.
        c = self.grids[self.nlevels-1].get_var("coeffs")
        c.v()[:,:] = coeffs.v()[:,:]

        self.grids[self.nlevels-1].fill_BC("coeffs")

        # put the coefficients on edges
        self.edge_coeffs.insert(0, ec.EdgeCoeffs(self.grids[self.nlevels-1].grid, c))

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
            self.edge_coeffs.insert(0, self.edge_coeffs[0].restrict())  #_EdgeCoeffs(self.grids[n].grid, coeffs_c))

            # if we are periodic, then we should force the edge coefficents
            # to be periodic
            # if self.grids[n].BCs["coeffs"].xlb == "periodic":
            #     self.edge_coeffs[0].x[self.grids[n].grid.ihi+1,:] = \
            #         self.edge_coeffs[0].x[self.grids[n].grid.ilo,:]

            # if self.grids[n].BCs["coeffs"].ylb == "periodic":
            #     self.edge_coeffs[0].y[:,self.grids[n].grid.jhi+1] = \
            #         self.edge_coeffs[0].y[:,self.grids[n].grid.jlo]

            n -= 1

    # FIXME: get rid of fortran testing stuff
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

            _v = mg_f.smooth_f(myg.qx, myg.qy, myg.ng,
                          nsmooth, np.asfortranarray(v.d), np.asfortranarray(f.d), bcints, np.asfortranarray(eta_x), np.asfortranarray(eta_y))

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


                    #CHANGED: This has been running slow, so have gone back to #old style indexing

                    #denom = (eta_x.ip_jp(1+ix, iy, s=2) + eta_x.ip_jp(ix, iy, s=2) +
                    #         eta_y.ip_jp(ix, 1+iy, s=2) + eta_y.ip_jp(ix, iy, s=2) )

                    #v.ip_jp(ix, iy, s=2)[:,:] = ( -f.ip_jp(ix, iy, s=2) +
                        # eta_{i+1/2,j} phi_{i+1,j}
                    #    eta_x.ip_jp(1+ix, iy, s=2) * v.ip_jp(1+ix, iy, s=2) +
                        # eta_{i-1/2,j} phi_{i-1,j}
                    #    eta_x.ip_jp(ix, iy, s=2) * v.ip_jp(-1+ix, iy, s=2) +
                        # eta_{i,j+1/2} phi_{i,j+1}
                    #    eta_y.ip_jp(ix, 1+iy, s=2) * v.ip_jp(ix, 1+iy, s=2) +
                        # eta_{i,j-1/2} phi_{i,j-1}
                    #    eta_y.ip_jp(ix, iy, s=2) * v.ip_jp(ix, -1+iy, s=2) ) / denom

                    denom = (
                        eta_x[myg.ilo+1+ix:myg.ihi+2:2,
                              myg.jlo+iy  :myg.jhi+1:2] +
                        #
                        eta_x[myg.ilo+ix  :myg.ihi+1:2,
                              myg.jlo+iy  :myg.jhi+1:2] +
                        #
                        eta_y[myg.ilo+ix  :myg.ihi+1:2,
                              myg.jlo+1+iy:myg.jhi+2:2] +
                        #
                        eta_y[myg.ilo+ix  :myg.ihi+1:2,
                              myg.jlo+iy  :myg.jhi+1:2])

                    v.d[myg.ilo+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi+1:2] = (
                        -f.d[myg.ilo+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi+1:2] +
                        # eta_{i+1/2,j} phi_{i+1,j}
                        eta_x[myg.ilo+1+ix:myg.ihi+2:2,
                              myg.jlo+iy  :myg.jhi+1:2] *
                        v.d[myg.ilo+1+ix:myg.ihi+2:2,
                          myg.jlo+iy  :myg.jhi+1:2] +
                        # eta_{i-1/2,j} phi_{i-1,j}
                        eta_x[myg.ilo+ix:myg.ihi+1:2,
                              myg.jlo+iy:myg.jhi+1:2]*
                        v.d[myg.ilo-1+ix:myg.ihi  :2,
                          myg.jlo+iy  :myg.jhi+1:2] +
                        # eta_{i,j+1/2} phi_{i,j+1}
                        eta_y[myg.ilo+ix:myg.ihi+1:2,
                              myg.jlo+1+iy:myg.jhi+2:2]*
                        v.d[myg.ilo+ix  :myg.ihi+1:2,
                          myg.jlo+1+iy:myg.jhi+2:2] +
                        # eta_{i,j-1/2} phi_{i,j-1}
                        eta_y[myg.ilo+ix:myg.ihi+1:2,
                              myg.jlo+iy:myg.jhi+1:2]*
                        v.d[myg.ilo+ix  :myg.ihi+1:2,
                          myg.jlo-1+iy:myg.jhi  :2]) / denom

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



    def _compute_residual(self, level):
        """ compute the residual and store it in the r variable"""

        v = self.grids[level].get_var("v").d
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")

        myg = self.grids[level].grid

        eta_x = self.edge_coeffs[level].x.d
        eta_y = self.edge_coeffs[level].y.d

        # compute the residual
        # r = f - L_eta phi
        """
        Gone back to old version as faster
        L_eta_phi = (
            # eta_{i+1/2,j} (phi_{i+1,j} - phi_{i,j})
            eta_x.ip(1)*(v.ip(1) - v.v()) - \
            # eta_{i-1/2,j} (phi_{i,j} - phi_{i-1,j})
            eta_x.v()*(v.v() - v.ip(-1)) + \
            # eta_{i,j+1/2} (phi_{i,j+1} - phi_{i,j})
            eta_y.jp(1)*(v.jp(1) - v.v()) - \
            # eta_{i,j-1/2} (phi_{i,j} - phi_{i,j-1})
            eta_y.v()*(v.v() - v.jp(-1)) )
        """

        L_eta_phi = (
            # eta_{i+1/2,j} (phi_{i+1,j} - phi_{i,j})
            eta_x[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] *
            (v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             v[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]) -
            # eta_{i-1/2,j} (phi_{i,j} - phi_{i-1,j})
            eta_x[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] *
            (v[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] -
             v[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1]) +
            # eta_{i,j+1/2} (phi_{i,j+1} - phi_{i,j})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] *
            (v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -  # y-diff
             v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1]) -
            # eta_{i,j-1/2} (phi_{i,j} - phi_{i,j-1})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] *
            (v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] -
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ]))

        r.v()[:,:] = f.v() - L_eta_phi
