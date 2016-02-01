from __future__ import print_function

import numpy as np
import mesh.metric as metric
import mesh.patch as patch


class BCObject_Sph(patch.BCObject):
    """Boundary condition container -- hold the BCs on each boundary
    for a single variable.

    For Neumann and Dirichlet BCs, a function callback can be stored
    for inhomogeous BCs.  This function should provide the value on
    the physical boundary (not cell center).  This is evaluated on the
    relevant edge when the __init__ routine is called.  For this
    reason, you need to pass in a grid object.  Note: this only
    ensures that the first ghost cells is consistent with the BC
    value.

    """

class Grid2d_Sph(patch.Grid2d):
    """
    the 2-d grid class.  The grid object will contain the coordinate
    information (at various centerings).

    A basic (1-d) representation of the layout is:

    |     |      |     X     |     |      |     |     X     |      |     |
    +--*--+- // -+--*--X--*--+--*--+- // -+--*--+--*--X--*--+- // -+--*--+
       0          ng-1    ng   ng+1         ... ng+nx-1 ng+nx      2ng+nx-1

                         ilo                      ihi

    |<- ng guardcells->|<---- nx interior zones ----->|<- ng guardcells->|

    The '*' marks the data locations.
    """

    def __init__(self, nx, ny, ng=1,
                 xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, R=0.0):
        """
        Create a Grid2d_Sph object.

        The only data that we require is the number of points that
        make up the mesh in each direction.  Optionally we take the
        extrema of the domain (default is [0,1]x[0,1]) and number of
        ghost cells (default is 1).

        Note that the Grid2d_Sph object only defines the discretization,
        it does not know about the boundary conditions, as these can
        vary depending on the variable.

        Parameters
        ----------
        nx : int
            Number of zones in the x-direction
        ny : int
            Number of zones in the y-direction
        ng : int, optional
            Number of ghost cells
        xmin : float, optional
            Physical coordinate at the lower x boundary
        xmax : float, optional
            Physical coordinate at the upper x boundary
        ymin : float, optional
            Physical coordinate at the lower y boundary
        ymax : float, optional
            Physical coordinate at the upper y boundary
        """

        super(Grid2d_Sph, self).__init__(nx, ny, ng=ng,
                      xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        self.R = R
        self.r2d = self.y2d + self.R
        self.r2v = self.r2d[self.ilo:self.ihi+1, self.jlo:self.jhi+1]

        self.metric = None

    def initialise_metric(self, rp, alpha, beta, gamma, cartesian=True):
        """
        Initialise metric. This is a separate function as only need to do it for relativistic systems.
        """
        self.metric = metric.Metric(self, rp, alpha, beta, gamma, cartesian)


    def norm(self, d):
        """
        find the norm of the quantity d defined on the same grid, in the
        domain's valid region
        """
        return np.sqrt(self.dx*self.dy*
                       np.sum((d[self.ilo:self.ihi+1,self.jlo:self.jhi+1]**2 * self.r2v).flat))


    def coarse_like(self, N):
        """
        return a new grid object coarsened by a factor n, but with
        all the other properties the same
        """
        g = Grid2d_Sph(self.nx/N, self.ny/N, ng=self.ng,
                      xmin=self.xmin, xmax=self.xmax,
                      ymin=self.ymin, ymax=self.ymax, R=self.R)

        if self.metric is not None:
            g.initialise_metric(self.metric.rp, self.metric.alpha, self.metric.beta, self.metric.gamma, self.metric.cartesian)

        return g


    def fine_like(self, N):
        """
        return a new grid object finer by a factor n, but with
        all the other properties the same
        """
        g = Grid2d_Sph(self.nx*N, self.ny*N, ng=self.ng,
                      xmin=self.xmin, xmax=self.xmax,
                      ymin=self.ymin, ymax=self.ymax, R=self.R)

        if self.metric is not None:
            g.initialise_metric(self.metric.rp, self.metric.alpha, self.metric.beta, self.metric.gamma, self.metric.cartesian)

        return g


class CellCenterData2d_Sph(patch.CellCenterData2d):
    """
    A class to define cell-centered data that lives on a grid.  A
    CellCenterData2d object is built in a multi-step process before
    it can be used.

    -- Create the object.  We pass in a grid object to describe where
       the data lives:

       my_data = patch.CellCenterData2d(myGrid)

    -- Register any variables that we expect to live on this patch.
       Here BCObject describes the boundary conditions for that variable.

       my_data.register_var('density', BCObject)
       my_data.register_var('x-momentum', BCObject)
       ...

    -- Register any auxillary data -- these are any parameters that are
       needed to interpret the data outside of the simulation (for
       example, the gamma for the equation of state).

       my_data.set_aux(keyword, value)

    -- Finish the initialization of the patch

       my_data.create()

    This last step actually allocates the storage for the state
    variables.  Once this is done, the patch is considered to be
   locked.  New variables cannot be added.
    """

    def restrict(self, varname):
        """
        Restrict the variable varname to a coarser grid (factor of 2
        coarser) and return an array with the resulting data (and same
        number of ghostcells)
        """

        fG = self.grid
        fData = self.get_var(varname)

        # account for the fact that the cells at higher j are slightly
        # bigger
        alpha  = (fG.R + 1.5 * fG.dy) / (fG.R + 0.5 * fG.dy)

        # allocate an array for the coarsely gridded data
        cG = fG.coarse_like(2)
        cData = cG.scratch_array()

        # fill the coarse array with the restricted data -- just
        # average the 4 fine cells into the corresponding coarse cell
        # that encompasses them.
        cData.v()[:,:] = 0.5 * (
            fData.v(s=2) + fData.ip(1, s=2) +
                  alpha * (fData.jp(1, s=2) + fData.ip_jp(1, 1, s=2))) / (1. + alpha)

        return cData


    def prolong(self, varname):
        """
        Prolong the data in the current (coarse) grid to a finer
        (factor of 2 finer) grid.  Return an array with the resulting
        data (and same number of ghostcells).  Only the data for the
        variable varname will be operated upon.

        We will reconstruct the data in the zone from the
        zone-averaged variables using the same limited slopes as in
        the advection routine.  Getting a good multidimensional
        reconstruction polynomial is hard -- we want it to be bilinear
        and monotonic -- we settle for having each slope be
        independently monotonic:

                  (x)         (y)
        f(x,y) = m    x/dx + m    y/dy + <f>

        where the m's are the limited differences in each direction.
        When averaged over the parent cell, this reproduces <f>.

        Each zone's reconstrution will be averaged over 4 children.

        +-----------+     +-----+-----+
        |           |     |     |     |
        |           |     |  3  |  4  |
        |    <f>    | --> +-----+-----+
        |           |     |     |     |
        |           |     |  1  |  2  |
        +-----------+     +-----+-----+

        We will fill each of the finer resolution zones by filling all
        the 1's together, using a stride 2 into the fine array.  Then
        the 2's and ..., this allows us to operate in a vector
        fashion.  All operations will use the same slopes for their
        respective parents.

        """

        cG = self.grid
        cData = self.get_var(varname)

        # allocate an array for the finely gridded data
        fG = cG.fine_like(2)
        fData = fG.scratch_array()

        # account for the fact that the cells at higher j are slightly
        # bigger
        # alpha  = (fG.R + 1.5 * fG.dy) / (fG.R + 0.5 * fG.dy)

        # slopes for the coarse data
        m_x = cG.scratch_array()
        m_x.v()[:,:] = 0.5*(cData.ip(1) - cData.ip(-1))

        m_y = cG.scratch_array()
        m_y.v()[:,:] = 0.5*(cData.jp(1) - cData.jp(-1))

        # fill the children
        fData.v(s=2)[:,:] = cData.v() - 0.25*m_x.v() - 0.25*m_y.v()     # 1 child
        fData.ip(1, s=2)[:,:] = cData.v() + 0.25*m_x.v() - 0.25*m_y.v() # 2
        fData.jp(1, s=2)[:,:] = cData.v() - 0.25*m_x.v() + 0.25*m_y.v() # 3
        fData.ip_jp(1, 1, s=2)[:,:] = cData.v() + 0.25*m_x.v() + 0.25*m_y.v() # 4

        return fData


    # NOTE: this is rather hacky - needed a method to impose BCs on data
    # which is not stored in the CCData2d object but shares the same
    # grid.
    def fill_BC_given_data(self, data, bcs):
        """
        Fill the boundary conditions.  This operates on a single state
        variable at a time, to allow for maximum flexibility.

        We do periodic, reflect-even, reflect-odd, and outflow

        Parameters
        ----------
        name : str
            The name of the variable for which to fill the BCs.

        """

        # there is only a single grid, so every boundary is on
        # a physical boundary (except if we are periodic)

        # Note: we piggy-back on outflow and reflect-odd for
        # Neumann and Dirichlet homogeneous BCs respectively, but
        # this only works for a single ghost cell


        # -x boundary
        if bcs[0] in ["outflow", "neumann"]:
            data[:self.grid.ilo,:] =  data[self.grid.ilo,np.newaxis,:]

        elif bcs[0] == "reflect-even":
            data[:self.grid.ilo,:] = data[2*self.grid.ng-1:2*self.grid.ng-self.grid.ilo-1:-1,:]

        elif bcs[0] in ["reflect-odd", "dirichlet"]:
            data[:self.grid.ilo,:] = -data[2*self.grid.ng-1:2*self.grid.ng-self.grid.ilo-1:-1,:]

        elif bcs[0] == "periodic":
            data[ :self.grid.ilo,:] = data[self.grid.ihi-self.grid.ng+1:self.grid.ihi+1,:]


        # +x boundary
        if bcs[1] in ["outflow", "neumann"]:
            data[self.grid.ihi+1:,:] = data[self.grid.ihi,np.newaxis,:]

        elif bcs[1] == "reflect-even":
            data[self.grid.ihi+1:,:] = data[self.grid.ihi:self.grid.ihi-self.grid.ng:-1,:]

        elif bcs[1] in ["reflect-odd", "dirichlet"]:
            data[self.grid.ihi+1:,:] = -data[self.grid.ihi:self.grid.ihi-self.grid.ng:-1,:]

        elif bcs[1] == "periodic":
            data[self.grid.ihi+1:,:] = data[self.grid.ng:2*self.grid.ng,:]


        # -y boundary
        if bcs[2] in ["outflow", "neumann"]:
            data[:,:self.grid.jlo] = data[:,self.grid.jlo,np.newaxis]


        elif bcs[2] == "reflect-even":
            data[:,:self.grid.jlo] = data[:,2*self.grid.ng-1:2*self.grid.ng-self.grid.jlo-1:-1]

        elif bcs[2] in ["reflect-odd", "dirichlet"]:
            data[:,:self.grid.jlo] = -data[:,2*self.grid.ng-1:2*self.grid.ng-self.grid.jlo-1:-1]

        elif bcs[2] == "periodic":
            data[:,:self.grid.jlo] = data[:,self.grid.jhi-self.grid.ng+1:self.grid.jhi+1]

        # +y boundary
        if bcs[3] in ["outflow", "neumann"]:

            data[:,self.grid.jhi+1:] = data[:,self.grid.jhi,np.newaxis]

        elif bcs[3] == "reflect-even":
            data[:,self.grid.jhi+1:] = data[:,self.grid.jhi-self.grid.ng+1:self.grid.jhi+1]

        elif bcs[3] in ["reflect-odd", "dirichlet"]:
            data[:,self.grid.jhi+1:] = -data[:,self.grid.jhi-self.grid.ng+1:self.grid.jhi+1]

        elif bcs[3] == "periodic":
            data[:,self.grid.jhi+1:] = data[:,:self.grid.jlo+1]
