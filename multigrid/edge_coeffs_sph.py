from multigrid.edge_coeffs import *
import numpy as np
from functools import partial

class EdgeCoeffsSpherical(EdgeCoeffs):
    """
    a simple container class to hold edge-centered coefficients
    and restrict them to coarse levels
    """
    def __init__(self, g, eta, empty=False):

        self.grid = g
        self.alphasq = g.scratch_array()
        if isinstance(g.metric.alpha, partial):
            self.alphasq.d[:,:] = g.metric.alpha(g).d2d()**2
        else:
            self.alphasq.d[:,:] = g.metric.alpha.v2df(g.qx, buf=1)**2

        if not empty:
            eta_x = g.scratch_array()
            eta_y = g.scratch_array()

            # the eta's are defined on the interfaces, so
            # eta_x[i,j] will be eta_{i-1/2,j} and
            # eta_y[i,j] will be eta_{i,j-1/2}

            b = (0,1)

            eta_x.v(buf=b)[:,:] = 0.5 * (
                eta.ip(-1, buf=b) * self.alphasq.ip(-1, buf=b) * np.sin(g.x2d[g.ilo-1:g.ihi+1,g.jlo:g.jhi+2])/ g.r2d[g.ilo-1:g.ihi+1,g.jlo:g.jhi+2]**3 +
                eta.v(buf=b) * self.alphasq.v(buf=b) * np.sin(g.x2d[g.ilo:g.ihi+2,g.jlo:g.jhi+2])/ g.r2d[g.ilo:g.ihi+2,g.jlo:g.jhi+2]**3)
            eta_y.v(buf=b)[:,:] = 0.5 * (
                eta.jp(-1, buf=b) * self.alphasq.jp(-1, buf=b) * g.r2d[g.ilo:g.ihi+2,g.jlo-1:g.jhi+1]**2 +
                eta.v(buf=b) * self.alphasq.v(buf=b) * g.r2d[g.ilo:g.ihi+2,g.jlo:g.jhi+2]**2)

            eta_x /= g.dx
            eta_y /= g.dy

            self.x = eta_x
            self.y = eta_y


    def restrict(self):
        """
        restrict the edge values to a coarser grid.  Return a new
        EdgeCoeffs object
        """

        fg = self.grid
        cg = self.grid.coarse_like(2)

        alphasq = fg.scratch_array()
        if isinstance(fg.metric.alpha, partial):
            alphasq.d[:,:] = fg.metric.alpha(fg).d2d()**2
        else:
            alphasq.d[:,:] = fg.metric.alpha.v2df(fg.qx, buf=1)**2
        alphasq.d[:,:] = fg.metric.alpha(fg).d2d()**2

        c_edge_coeffs = EdgeCoeffsSpherical(cg, None, empty=True)

        c_eta_x = cg.scratch_array()
        c_eta_y = cg.scratch_array()

        b = (0, 1, 0, 0)
        #c_eta_x.v(buf=b)[:,:] = 0.5*(self.x.v(buf=b, s=2) + self.x.jp(1, buf=b, s=2))

        c_eta_x.v(buf=b)[:,:] = 0.5 * (
            self.x.v(buf=b, s=2) * self.alphasq.v(buf=b, s=2) *
            np.sin(fg.x2d[fg.ilo:fg.ihi+2:2,fg.jlo:fg.jhi+1:2]) /
            fg.r2d[fg.ilo:fg.ihi+2:2,fg.jlo:fg.jhi+1:2]**3 +
            self.x.jp(1, buf=b, s=2) * self.alphasq.jp(1, buf=b, s=2) *
            np.sin(fg.x2d[fg.ilo:fg.ihi+2:2,fg.jlo+1:fg.jhi+2:2]) /
            fg.r2d[fg.ilo:fg.ihi+2:2,fg.jlo+1:fg.jhi+2:2]**3)


        b = (0, 0, 0, 1)
        #c_eta_y.v(buf=b)[:,:] = 0.5*(self.y.v(buf=b, s=2) + self.y.ip(1, buf=b, s=2))

        c_eta_y.v(buf=b)[:,:] = 0.5 * (
            self.y.v(buf=b, s=2) * self.alphasq.v(buf=b, s=2) *
            fg.r2d[fg.ilo:fg.ihi+1:2,fg.jlo:fg.ihi+2:2]**2 +
            self.y.ip(1, buf=b, s=2) * self.alphasq.ip(1, buf=b, s=2) *
            fg.r2d[fg.ilo+1:fg.ihi+2:2,fg.jlo:fg.ihi+2:2]**2)

        # redo the normalization
        # I don't think you need to include the r, sin(x) factors
        # as these are basically the same on the new grid?
        c_edge_coeffs.x = cg.scratch_array()
        c_edge_coeffs.x.d[:,:] = c_eta_x.d * fg.dx  / cg.dx

        c_edge_coeffs.y = cg.scratch_array()
        c_edge_coeffs.y.d[:,:] = c_eta_y.d * fg.dy /cg.dy

        return c_edge_coeffs
