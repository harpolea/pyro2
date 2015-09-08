"""
TODO: see if can make this a singleton class for the case where the metric is time-independent.
"""

from __future__ import print_function

import numpy as np
import sys
from util import msg
from lm_gr.problems import *


class Metric:

    def __init__(self, cellData, rp, alpha, beta, gamma):
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

        self.cc_data = cellData
        self.rp = rp
        self.alpha = alpha
        self.beta = beta
        self.W = 1.  # Lorentz factor.
        self.gamma = gamma
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
        myg = self.cc_data.grid
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

    def calcW(self, u=None, v=None):
        """
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
        myg = self.cc_data.grid
        W = myg.scratch_array()
        if u is None:
            u = self.cc_data.get_var("x-velocity")
        if v is None:
            v = self.cc_data.get_var("y-velocity")
        c = self.rp.get_param("lm-gr.c")

        # for loop here as otherwise my brain hurts
        # FIXME: I think the for-loop here might be seriously slowing stuff down as u0 is calculated quite a lot. Do this with slicing.
        for i in range(self.cc_data.grid.qx):
            for j in range(self.cc_data.grid.qy):
                V = np.array([u.d[i,j], v.d[i,j]]) + self.beta
                # set W = V^i*V_i = V^i * gamma_ij * V^j
                # TODO: do this with numpy einsum
                W.d[i,j] = (np.mat(V) * np.mat(self.gamma[i,j,:,:]) * np.mat(V).T).item()
        W.d[:,:] = 1. - W.d / (self.alpha.d2d()**2 * c**2)

        try:
            W.d[:, :] = 1. / np.sqrt(W.d)
        except FloatingPointError:
            msg.bold('\nError!')
            print('Tried to take the square root of a negative Lorentz \
                  factor! \nTry checking your velocities?\n')
            print((u.d[-10:, -10:]**2 + v.d[-10:, -10:]**2)/c**2)
            print(u.d.min, v.d.min)
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
        myg = self.cc_data.grid
        u0 = myg.scratch_array()
        u0.d[:,:] = W.d / self.alpha.d2d()

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
        met[0, 0] = -self.alpha.d[x[2]]**2 + np.dot(self.beta, self.beta)
        met[0, 1:] = np.transpose(self.beta)
        met[1:, 0] = self.beta
        met[1:, 1:] = self.gamma[x[1], x[2], :, :]

        return met

    def christoffels(self, x):
        """
        Calculates the Christoffel symbols of the metric at the given point.

        TODO: un-hardcode these

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

        # For simple time-lagged metric, only have 7 non-zero (4 unique) christoffels.
        # t_tr
        christls[0, 0, 2] = g / (self.alpha.d[x[2]]**2 * c**2 * R)
        # t_rt
        christls[0, 2, 0] = christls[0, 0, 2]
        # r_tt
        christls[2, 0, 0] = g * self.alpha.d[x[2]]**2 / (c**2 * R)
        # r_xx
        christls[2, 1, 1] = g / (c**2 * R * self.alpha.d[x[2]]**2)
        # r_rr
        christls[2, 2, 2] = -christls[2, 1, 1]
        # x_xr
        christls[1, 1, 2] = christls[2, 2, 2]
        # x_rx
        christls[1, 2, 1] = christls[2, 2, 2]

        # For non-simple, we have to do more icky stuff including time and
        # space
        # derivatives of stuff, so I shall not do this for now.

        return christls