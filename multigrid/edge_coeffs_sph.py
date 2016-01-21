"""
"""
from multigrid.edge_coeffs import *
import numpy as np

class EdgeCoeffsSpherical(EdgeCoeffs):
    """
    a simple container class to hold edge-centered coefficients
    and restrict them to coarse levels
    """
    def __init__(self, g, eta, empty=False):

        self.grid = g

        if not empty:
            eta_x = g.scratch_array()
            eta_y = g.scratch_array()

            # the eta's are defined on the interfaces, so
            # eta_x[i,j] will be eta_{i-1/2,j} and
            # eta_y[i,j] will be eta_{i,j-1/2}

            b = (0,1)

            eta_x.v(buf=b)[:,:] = 0.5*(eta.ip(-1, buf=b) + eta.v(buf=b))
            eta_y.v(buf=b)[:,:] = 0.5*(eta.jp(-1, buf=b) + eta.v(buf=b))

            eta_x *= np.sin(g.x2d - 0.5*g.dx) / g.dx
            eta_y *= (g.r2d - 0.5*g.dy)**2 / g.dy

            self.x = eta_x
            self.y = eta_y


    def restrict(self):
        """
        restrict the edge values to a coarser grid.  Return a new
        EdgeCoeffs object
        """

        cg = self.grid.coarse_like(2)

        c_edge_coeffs = EdgeCoeffsSpherical(cg, None, empty=True)

        c_eta_x = cg.scratch_array()
        c_eta_y = cg.scratch_array()

        fg = self.grid

        b = (0, 1, 0, 0)
        c_eta_x.v(buf=b)[:,:] = 0.5*(self.x.v(buf=b, s=2) + self.x.jp(1, buf=b, s=2))

        # coarsen r and x
        c_r_x = cg.scratch_array()
        c_x_x = cg.scratch_array()
        c_r_x.d[:-1,:-1] = 0.5 * (fg.r2d[::2,::2] + fg.r2d[1::2,::2])
        c_x_x.d[:-1,:-1] = 0.5 * (fg.x2d[::2,::2] + fg.x2d[1::2,::2])

        b = (0, 0, 0, 1)
        c_eta_y.v(buf=b)[:,:] = 0.5*(self.y.v(buf=b, s=2) + self.y.ip(1, buf=b, s=2))

        # coarsen r
        c_r_y = cg.scratch_array()
        c_r_y.d[:-1,:-1] = 0.5 * (fg.r2d[::2,::2] + fg.r2d[::2,1::2])

        # redo the normalization
        mask = (c_x_x.d > 0.)
        c_edge_coeffs.x = cg.scratch_array()
        c_edge_coeffs.x.d[mask] = c_eta_x.d[mask] * fg.dx * np.sin(cg.x2d[mask] - 0.5*cg.dx) / (cg.dx * np.sin(c_x_x.d[mask] - 0.5*fg.dx))

        mask = (c_x_x.d > 0.)
        c_edge_coeffs.y = cg.scratch_array()
        c_edge_coeffs.y.d[mask] = c_eta_y.d[mask] * fg.dy * (cg.r2d[mask] - 0.5*cg.dy)**2 /(cg.dy * (c_r_y.d[mask] - 0.5*fg.dy)**2)

        return c_edge_coeffs
