"""
TODO: see if can make this a singleton class for the case where the metric is time-independent.

NOTE: in future, might want to inherit from base Metric class and
implement everything separately for different subclasses. In this case,
alpha and beta will be defined in the class itself rather than passed
in to the constructor.
"""

from __future__ import print_function

import numpy as np
import sys
from util import msg
import traceback

class Metric:

    def __init__(self, grid, rp, alpha, beta, gamma, cartesian=True):
        """
        Initialize the Metric object. This is a standard 2+1 metric

        Parameters
        ----------
        cellData : CellCenterData2d object
            Simulation data object
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        alpha : float array
            lapse function
        beta : float array
            shift vector
        gamma : float array
            spatial 2-metric
        """

        self.grid = grid
        self.rp = rp
        self.alpha = alpha
        self.beta = beta
        self.W = 1.  # Lorentz factor.
        self.gamma = gamma
        self.cartesian = cartesian

        # store christoffels here as for time-independent metric these shall not change.
        myg = self.grid
        self.chrls = np.array([[self.christoffels([0, i, j])
                           for j in range(myg.qy)] for i in range(myg.qx)])

        #self.inverse = self.inverse_metric()

        np.seterr(invalid='raise')  # raise numpy warnings as errors

    def dets(self):
        """
        Calculates the square roots of the 2- and 2+1-metric determinants and
        returns them.

        Returns
        -------
        sg, sgamma : float array
            square roots of the 2- and 2+1-metric determinants on grid
        """
        myg = self.grid
        sg = myg.scratch_array()
        sgamma = myg.scratch_array()

        # calculate metric at point then take square roots of determinants.
        sg.d[:, :] = [[np.sqrt(-1. * np.linalg.det(self.g([0, i, j])))
                      for j in range(0, myg.qy)] for i in range(0, myg.qx)]

        # TODO: check that this is equal to alpha * sgamma

        sgamma.d[:, :] = [[np.sqrt(np.linalg.det((self.g([0, i, j]))[1:, 1:]))
                           for j in range(0, myg.qy)]
                          for i in range(0, myg.qx)]

        return sg, sgamma

    def inverse_metric(self):
        r"""
        Returns components of the inverse metric, assuming it's diagonal
        for now.
        """
        myg = self.grid
        alpha = self.alpha(myg)
        if self.cartesian:
            return np.array([[np.diag([-1./alpha.d[i,j]**2, alpha.d[i,j]**2, alpha.d[i,j]**2]) for j in range(0,myg.qy)] for i in range(0, myg.qx)])
        else:
            return np.array([[np.diag([-1./alpha.d[i,j]**2, alpha.d[i,j]**2, alpha.d[i,j]**2/myg.r2d[i,j]]) for j in range(0,myg.qy)] for i in range(0, myg.qx)])

    # cannot use numba here as it doesn't support list comprehensions :(
    def calcW(self, u=None, v=None):
        r"""
        Calculates the Lorentz factor and returns it.

        Variables
        ---------

        u, v :  x, r components of :math:`U^i`, where :math:`U^i = u^i / u^0`

        V :     contravariant components of the 3-velocity,
                ..math::

                    V^i = (U^i + beta^i) / alpha

        W :     Lorentz factor
                ..math::

                    W =  (1 - V^i*V_i)^(-1/2)

        Returns
        -------
        W : float array
            Lorentz factor on grid
        """
        myg = self.grid
        W = myg.scratch_array()
        if u is None:
            #u = self.cc_data.get_var("x-velocity")
            u = myg.scratch_array()
        if v is None:
            #v = self.cc_data.get_var("y-velocity")
            v = myg.scratch_array()
        c = self.rp.get_param("lm-gr.c")
        alpha = self.alpha(myg)
        gamma = self.gamma(myg)

        Vs = np.array([[np.array([u.d[i,j], v.d[i,j]]) + self.beta
            for j in range(myg.qy)] for i in range(myg.qx)])

        Vs[:,:,:] /= alpha.d2d()[:,:,np.newaxis]**2

        W.d[:,:] = np.array([[ np.inner(np.inner(Vs[i,j,:],
            gamma[i,j,:,:]), np.transpose(Vs[i,j,:]))
            for j in range(myg.qy)] for i in range(myg.qx)])

        W.d[:,:] = 1. - W.d / c**2

        try:
            W.d[:, :] = 1. / np.sqrt(W.d)
        except FloatingPointError:
            msg.bold('\nError!')
            print('Tried to take the square root of a negative Lorentz factor! \nTry checking your velocities?\n')
            print((u.d[-10:, -10:]**2 + v.d[-10:, -10:]**2)/c**2)
            print('umin: {}, umax: {}, vmin: {}, vmax:{}'.format(u.d.min(), u.d.max(), v.d.min(), v.d.max()))
            traceback.print_stack()
            np.set_printoptions(threshold=np.nan)
            #mag_vel = u.d**2 + v.d**2
            W.d[:,:] = np.array([[ np.inner(np.inner(Vs[i,j,:],
                gamma[i,j,:,:]), np.transpose(Vs[i,j,:]))
                for j in range(myg.qy)] for i in range(myg.qx)])
            print(W.d)
            sys.exit()

        return W

    def calcu0(self, u=None, v=None):
        """
        Calculates the timelike coordinate of the 2+1-velocity using the Lorentz
        factor and alpha, so W = alpha * u0

        Returns
        -------
        u0 : float array
            u0 on grid
        """

        W = self.calcW(u=u, v=v)
        myg = self.grid
        alpha = self.alpha(myg)
        u0 = myg.scratch_array(data=W.d/alpha.d2d())

        return u0

    def g(self, x):
        """
        Calculates the 2+1 downstairs metric at the coordinate x.
        Currently only alpha has any x-dependence (in radial direction only)

        Parameters
        ----------
        x : float array
            2+1-coordinate of point where g is to be calculated

        Returns
        -------
        met : float array
            (d+1)*(d+1) array containing metric
        """

        met = np.diag([-1., 1., 1.])  # flat default
        met[0, 0] = -self.alpha(self.grid).d[x[2]]**2 + np.dot(self.beta, self.beta)
        met[0, 1:] = np.transpose(self.beta)
        met[1:, 0] = self.beta
        met[1:, 1:] = self.gamma(self.grid)[x[1], x[2], :, :]

        return met

    def christoffels(self, x):
        """
        Calculates the Christoffel symbols of the metric at the given point.

        Parameters
        ----------
        x : float array
            2+1 coordinate of point where christoffels are to be calculated.

        Returns
        -------
        christls : float array
            (d+1)^3 array containing christoffel symbols at x
        """

        christls = np.zeros((3, 3, 3))

        # r = self.cc_data.grid.y[x[2]]
        g = self.rp.get_param("lm-gr.grav")
        R = self.rp.get_param("lm-gr.radius")
        c = self.rp.get_param("lm-gr.c")
        alpha = self.alpha(self.grid)

        if self.cartesian:
            # For simple time-lagged metric, only have 7 non-zero (4 unique) christoffels.
            # t_tr
            christls[0, 0, 2] = g / (alpha.d[x[2]]**2 * c**2 * R)
            # t_rt
            christls[0, 2, 0] = christls[0, 0, 2]
            # r_tt
            christls[2, 0, 0] = g * alpha.d[x[2]]**2 / (c**2 * R)
            # r_xx
            christls[2, 1, 1] = g / (c**2 * R * alpha.d[x[2]]**2)
            # r_rr
            christls[2, 2, 2] = -christls[2, 1, 1]
            # x_xr
            christls[1, 1, 2] = christls[2, 2, 2]
            # x_rx
            christls[1, 2, 1] = christls[2, 2, 2]
        else: # spherical polar
            # t_tr
            christls[0, 0, 2] = g / (alpha.d[x[2]]**2 * c**2 * R)
            # t_rt
            christls[0, 2, 0] = christls[0, 0, 2]
            # r_tt
            christls[2, 0, 0] = g * alpha.d[x[2]]**2 / (c**2 * R)
            # r_theta theta
            christls[2, 1, 1] = -R * alpha.d[x[2]]**2 * (1. + g/c**2)
            # r_rr
            christls[2, 2, 2] = -g * alpha.d[x[2]]**2 / (R * c**2)
            # theta_theta r
            christls[1, 1, 2] = alpha.d[x[2]]**2 * (1. + g/c**2) / R
            # theta_r theta
            christls[1, 2, 1] = christls[1, 1, 2]

        # For non-simple, we have to do more icky stuff including time and
        # space
        # derivatives of stuff, so I shall not do this for now.

        return christls
