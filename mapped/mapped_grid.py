"""
Stores and manages particles and updates their positions based
on the velocity on the grid.
"""

import numpy as np
import mesh.patch as patch


class MappedGrid2d(patch.Grid2d):
    """
    Mapped grid class
    """

    def __init__(self, cart_grid, hxmin=0.0, hxmax=1.0, hymin=0.0, hymax=1.0):

        super().__init__(cart_grid.nx, cart_grid.ny, cart_grid.ng,
                         cart_grid.xmin, cart_grid.xmax, cart_grid.ymin, cart_grid.ymax)

        self.cart = cart_grid

        self.hxmin, self.hxmax = hxmin, hxmax
        self.hymin, self.hymax = hymin, hymax

        self.dhx, self.dhy = self.dh()
        self.hx2d, self.hy2d = self.map_from_cart()

        self.cell_areas = self.cell_area()

    def norm(self, d):
        """
        find the norm of the quantity d defined on the same grid, in the
        domain's valid region
        """
        return np.sqrt(np.sum((self.cell_areas[self.ilo:self.ihi + 1, self.jlo:self.jhi + 1] *
                               d[self.ilo:self.ihi + 1, self.jlo:self.jhi + 1]**2).flat))

    def map_to_cart(self):
        """
        Define map from grid to cartesian
        """
        pass

    def map_from_cart(self):
        """
        Define map from cartesian to grid
        """
        pass

    def cell_area(self):
        """
        Calculate cell areas on non-cartesian grid.
        """
        pass

    def normals(self, dir):
        """
        Calculate the components of the edge normals.
        Dir is whether we want the x edges (0) or the y edges (1).

        Returns
        -------
        n_x, n_y
        """
        pass

    def dh(self):
        """
        Calculate edge lengths

        Returns
        -------
        dh_x, dh_y
        """
        dhx = self.cart.scratch_array()
        dhy = self.cart.scratch_array()

        dhx[:, :] = (self.hxmax - self.hxmin) / self.cart.nx

        dhy[:, :] = (self.hymax - self.hymin) / self.cart.ny

        return dhx, dhy


class Rectilinear(MappedGrid2d):
    """
    Mapped grid class
    """

    def map_to_cart(self):
        """
        Define map from grid to cartesian
        """
        pass

    def map_from_cart(self):
        """
        Define map from cartesian to grid
        """
        return self.cart.x2d * self.dhx / self.cart.dx, self.cart.y2d * self.dhy / self.cart.dy

    def cell_area(self):
        """
        Calculate cell areas on non-cartesian grid.
        """
        return self.dhx * self.dhy

    def normals(self, dir):
        """
        Calculate the components of the edge normals.
        Dir is whether we want the x edges (0) or the y edges (1).

        Returns
        -------
        n_x, n_y
        """
        n_x = self.cart.scratch_array()
        n_y = self.cart.scratch_array()

        if dir == 0:
            n_x[:, :] = 1
            n_y[:, :] = 0
        else:
            n_x[:, :] = 0
            n_y[:, :] = 1

        return n_x, n_y


class Curvilinear(MappedGrid2d):
    """
    Mapped grid class
    """

    def __init__(self, cart_grid, hxmin=0.0, hxmax=1.0, hymin=0.0, hymax=0.5 * np.pi):
        super().__init__(cart_grid, hxmin, hxmax, hymin, hymax)

        # rescale dhy by r
        self.dhy *= self.hx2d

        # aliases to make things easier
        self.dr = self.dhx
        self.dth = self.dhy
        self.r2d = self.hx2d
        self.theta2d = self.hy2d

    # def map_to_cart(self):
    #     """
    #     Define map from grid to cartesian
    #     """
    #     pass

    def map_from_cart(self):
        """
        Define map from cartesian to grid
        """

        r_range = np.repeat(np.arange(self.cart.qx), self.cart.qy)
        r_range.shape = (self.cart.qx, self.cart.qy)
        rl = (r_range - self.cart.ng) * self.dhx + self.hxmin
        rr = (np.arange(self.cart.qx) + 1.0 -
              self.cart.ng) * self.dhx + self.hxmin

        r = 0.5 * (rl + rr)
        #r = np.repeat(r, self.cart.qy)
        # r.shape = (self.cart.qx, self.cart.qy)

        th_range = np.repeat(np.arange(self.cart.qy), self.cart.qx)
        th_range.shape = (self.cart.qy, self.cart.qx)
        th_range = np.transpose(th_range)

        thl = (th_range - self.cart.ng) * self.dhy + self.hymin
        thr = (np.arange(self.cart.qy) + 1.0 -
               self.cart.ng) * self.dhy + self.hymin

        theta = 0.5 * (thl + thr)
        #theta = np.repeat(theta, self.qx)
        #theta.shape = (self.cart.qy, self.cart.qx)
        # theta = np.transpose(theta)

        return r, theta

    def cell_area(self):
        """
        Calculate cell areas on non-cartesian grid.
        """
        areas = self.cart.scratch_array()
        areas.v()[:, :] = 0.5 * self.dhy.v() * \
            (self.hx2d.ip(1)**2 - self.hx2d.v()**2)

        return areas

    def normals(self, dir):
        """
        Calculate the components of the edge normals.
        Dir is whether we want the x edges (0) or the y edges (1).

        Returns
        -------
        n_x, n_y
        """
        n_x = self.cart.scratch_array()
        n_y = self.cart.scratch_array()

        if dir == 0:
            n_x[:, :] = 1
            n_y[:, :] = 0
        else:
            n_x[:, :] = 0
            n_y[:, :] = 1

        return n_x, n_y

    def dh(self):
        """
        Calculate edge lengths

        Returns
        -------
        dh_x, dh_y
        """
        dhx = self.cart.scratch_array()
        dhy = self.cart.scratch_array()

        dhx[:, :] = (self.hxmax - self.hxmin) / self.cart.nx

        dhy[:, :] = (self.hymax - self.hymin) / self.cart.ny

        return dhx, dhy


class StructuredData2d(patch.CellCenterData2d):
    pass
