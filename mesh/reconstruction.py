"""
this library implements the limiting functions used in the
! reconstruction.  We use F90 for both speed and clarity, since
! the numpy array notations can sometimes be confusing.
"""

import math

import numpy as np

#-----------------------------------------------------------------------------
# nolimit
#-----------------------------------------------------------------------------
def nolimit(idir, a, myg):

    # just to a centered difference -- no limiting

    #initialise some stuff
    lda = np.zeros((myg.qx,myg.qy), dtype=np.float64)


    # call this as: lda = reconstruction_f.nolimit(1,a,qx,qy,ng)

    if idir==1:
        lda[myg.ilo-2: myg.ihi+3, myg.jlo-2: myg.jhi+3] = 0.5 * \
            (a[myg.ilo-1:myg.ihi+4, myg.jlo-2:myg.jlo+3] - \
            a[myg.ilo-3:myg.ihi+2, myg.jlo-2:myg.jlo+3])

    else:
        lda[myg.ilo-2: myg.ihi+3, myg.jlo-2: myg.jhi+3] = 0.5 * \
            (a[myg.ilo-2:myg.ihi+3, myg.jlo-1:myg.jlo+4] - \
            a[myg.ilo-2:myg.ihi+3, myg.jlo-3:myg.jlo+2])

    return lda[:,:]




#----------------------------------------------------------------------------
# limit2
#-----------------------------------------------------------------------------
def limit2(idir, a, myg):

    # 2nd order limited centered difference

    #initialise some stuff
    lda = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    test = np.zeros((myg.qx,myg.qy), dtype=np.float64)



    if idir==1:

        # test whether we are at an extremum
        test[myg.ilo-3:myg.ihi+4, myg.jlo-3:myg.jhi+4] = \
            (a[myg.ilo-2:myg.ihi+5, myg.jlo-3:myg.jhi+4] - \
            a[myg.ilo-3:myg.ihi+4, myg.jlo-3:myg.jhi+4]) * \
            (a[myg.ilo-3:myg.ihi+4, myg.jlo-3:myg.jhi+4] - \
            a[myg.ilo-4:myg.ihi+3, myg.jlo-3:myg.jhi+4])


        for j in range(myg.jlo-3, myg.jhi+4):
            for i in range(myg.ilo-3, myg.ihi+4):

                if (test > 0.0):
                    lda[i,j] = min(0.5*abs(a[i+1,j] - a[i-1,j]), \
                        min(2.0*abs(a[i+1,j] - a[i,j]), \
                        2.0*abs(a[i,j] - a[i-1,j]))) * \
                        np.sign(1.0,a[i+1,j] - a[i-1,j])



    else:

        # test whether we are at an extremum
        test[myg.ilo-3:myg.ihi+4, myg.jlo-3:myg.jhi+4] = \
            (a[myg.ilo-3:myg.ihi+4, myg.jlo-2:myg.jhi+5] - \
            a[myg.ilo-3:myg.ihi+4, myg.jlo-3:myg.jhi+4]) * \
            (a[myg.ilo-3:myg.ihi+4, myg.jlo-3:myg.jhi+4] - \
            a[myg.ilo-3:myg.ihi+4, myg.jlo-4:myg.jhi+3])

        for j in range(myg.jlo-3, myg.jhi+4):
            for i in range(myg.ilo-3, myg.ihi+4):

                if (test > 0.):
                    lda[i,j] = min(0.5*abs(a[i,j+1] - a[i,j-1]), \
                        min(2.0*abs(a[i,j+1] - a[i,j]), \
                        2.0*abs(a[i,j] - a[i,j-1]))) * \
                        np.sign(1.0,a[i,j+1] - a[i,j-1])

    return lda[:,:]



#-----------------------------------------------------------------------------
# limit4
#-----------------------------------------------------------------------------
def limit4(idir, a, myg):

    """
    4th order limited centered difference

    See Colella (1985) Eq. 2.5 and 2.6, Colella (1990) page 191 (with
    the delta a terms all equal) or Saltzman 1994, page 156
    """

    #initialise some stuff
    lda = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    test = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    temp = np.zeros((myg.qx,myg.qy), dtype=np.float64)

    # first get the 2nd order estimate
    temp[:,:] = limit2(idir, a, myg)

    if idir ==1:
        # test whether we are at an extremum
        test[myg.ilo-2:myg.ihi+3, myg.jlo-2:myg.jhi+3] = \
            (a[myg.ilo-3:myg.ihi+4, myg.jlo-2:myg.jhi+3] - \
            a[myg.ilo-2:myg.ihi+3, myg.jlo-2:myg.jhi+3]) * \
            (a[myg.ilo-2:myg.ihi+3, myg.jlo-2:myg.jhi+3] - \
            a[myg.ilo-3:myg.ihi+2, myg.jlo-2:myg.jhi+3])


        for j in range(myg.jlo-2, myg.jhi+3):
            for i in range(myg.ilo-2, myg.ihi+3):

                if (test > 0.0):
                    lda[i,j] = \
                        min( (2./3.)*abs(a[i+1,j] - a[i-1,j] - \
                        0.25*(temp[i+1,j] + temp[i-1,j])), \
                        min(2.0*abs(a[i+1,j] - a[i  ,j]), \
                        2.0*abs(a[i  ,j] - a[i-1,j])) ) * \
                        np.sign(1.,a[i+1,j] - a[i-1,j])


    else:

        # test whether we are at an extremum
        test[myg.ilo-2:myg.ihi+3, myg.jlo-2:myg.jhi+3] = \
            (a[myg.ilo-2:myg.ihi+3, myg.jlo-1:myg.jhi+4] - \
            a[myg.ilo-2:myg.ihi+3, myg.jlo-2:myg.jhi+3]) * \
            (a[myg.ilo-2:myg.ihi+3, myg.jlo-2:myg.jhi+3] - \
            a[myg.ilo-2:myg.ihi+3, myg.jlo-3:myg.jhi+2])

        for j in range(myg.jlo-2, myg.jhi+3):
            for i in range(myg.ilo-2, myg.ihi+3):

                if (test > 0.):
                    lda[i,j] = \
                        min( (2./3.)*abs(a[i,j+1] - a[i,j-1] - \
                        0.25*(temp[i,j+1] + temp[i,j-1])), \
                        min(2.*abs(a[i,j+1] - a[i,j  ]), \
                        2.*abs(a[i,j  ] - a[i,j-1])) ) * \
                        np.sign(1.,a[i,j+1] - a[i,j-1])

    return lda[:,:]



#-----------------------------------------------------------------------------
# flatten
#-----------------------------------------------------------------------------
def flatten(idir, p, u, myg, smallp, delta, z0, z1):

    """
    ! 1-d flattening near shocks
    !
    ! flattening kicks in behind strong shocks and reduces the
    ! reconstruction to using piecewise constant slopes, making things
    ! first-order.  See Saltzman (1994) page 159 for this
    ! implementation.
    """

    #initialise some stuff
    xi = np.ones((myg.qx,myg.qy), dtype=np.float64)
    test1 = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    test2 = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    dp = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    dp2 = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    z = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    oness = np.ones((myg.qx,myg.qy), dtype=np.float64)


    if idir==1:

        dp[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            abs(p[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] - \
            p[myg.ilo-3:myg.ihi+2,myg.jlo-2:myg.jhi+3])

        dp2[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            abs(p[myg.ilo:myg.ihi+5,myg.jlo-2:myg.jhi+3] - \
            p[myg.ilo-4:myg.ihi+1,myg.jlo-2:myg.jhi+3])

        z[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            dp[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] / \
            np.maximum(dp2[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3], \
            smallp * oness[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3])

        test1[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            u[myg.ilo-3:myg.ihi+2,myg.jlo-2:myg.jhi+3] - \
            u[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3]

        test2[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            dp / np.minimum(p[myg.ilo-3:myg.ihi+2,myg.jlo-2:myg.jhi+3], \
            p[myg.ilo-3:myg.ihi+2,myg.jlo-2:myg.jhi+3])


        for j in range(myg.jlo-2, myg.jhi+3):
            for i in range(myg.ilo-2, myg.ihi+3):

                if (test1 > 0. and test2 > delta):
                    xi[i,j] = min(1., max(0., 1. - (z - z0)/(z1 - z0)))



    else:

        dp[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            abs(p[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] - \
            p[myg.ilo-2:myg.ihi+3,myg.jlo-3:myg.jhi+2])

        dp2[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            abs(p[myg.ilo-2:myg.ihi+3,myg.jlo:myg.jhi+5] - \
            p[myg.ilo-2:myg.ihi+3,myg.jlo-4:myg.jhi+1])

        z[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            dp[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] / \
            np.maximum(dp2[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3], \
            smallp * oness[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3])

        test1[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            u[myg.ilo-2:myg.ihi+3,myg.jlo-3:myg.jhi+2] - \
            u[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4]

        test2[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
            dp / np.minimum(p[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4], \
            p[myg.ilo-2:myg.ihi+3,myg.jlo-3:myg.jhi+2])

        for j in range(myg.jlo-2, myg.jhi+3):
            for i in range(myg.ilo-2, myg.ihi+3):

                if (test1 > 0. and test2 > delta):
                    xi[i,j] = min(1., max(0., 1. - (z - z0)/(z1 - z0)))


    return xi[:,:]


#-----------------------------------------------------------------------------
# flatten_multid
#-----------------------------------------------------------------------------
def flatten_multid(xi_x, xi_y, p, myg):

    # multi-dimensional flattening

    #initialise some stuff
    xi = np.ones((myg.qx,myg.qy), dtype=np.float64)


    for j in range(myg.jlo-2, myg.jhi+3):
        for i in range(myg.ilo-2, myg.ihi+3):

            sx = int(np.sign(1., p[i+1,j] - p[i-1,j]))
            sy = int(np.sign(1., p[i,j+1] - p[i,j-1]))

            xi[i,j] = min(min(xi_x[i,j], xi_x[i-sx,j]),\
            min(xi_y[i,j], xi_y[i,j-sy]))

    return xi[:,:]
