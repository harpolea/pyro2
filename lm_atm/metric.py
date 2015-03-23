from __future__ import print_function

import numpy as np
import sys
from util import msg
from lm_atm.problems import *


class Metric:

    def __init__(self, cellData, rp, alpha, beta, gamma):
        """
        Initialize the Metric object. This is a standard 3+1 metric.

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
            spatial 3-metric
        """


        self.cc_data = cellData
        self.rp = rp
        self.alpha = alpha
        self.beta = beta
        self.W = 1. #Lorentz factor.
        self.gamma = gamma
        np.seterr(invalid='raise') #raise numpy warnings as errors





    def dets(self):
        """
        Calculates the square roots of the 3- and 4-metric determinants and
        returns them.

        Returns
        -------
        sg, sgamma : float array
            square roots of the 3- and 4-metric determinants on grid
        """

        sg = self.cc_data.grid.scratch_array()
        sgamma = self.cc_data.grid.scratch_array()

        #calculate metric at point then take square roots of determinants.
        sg[:,:] = [[np.sqrt(-1.*np.linalg.det(self.g([0,i,j]))) \
            for j in range(0, self.cc_data.grid.qy)] \
            for i in range(0, self.cc_data.grid.qx)]

        sgamma[:,:] = [[np.sqrt(np.linalg.det((self.g([0,i,j]))[1:,1:])) \
            for j in range(0, self.cc_data.grid.qy)] \
            for i in range(0, self.cc_data.grid.qx)]

        return sg, sgamma




    def calcW(self):
        """
        Calculates the Lorentz factor and returns it.

        Returns
        -------
        W : float array
            Lorentz factor on grid
        """

        W = self.cc_data.grid.scratch_array()
        #W = np.ones(np.shape(W))

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        c = self.rp.get_param("lm-atmosphere.c")

        W[:,:] = 1. - (u[:,:]**2 + v[:,:]**2)/c**2

        try:
            W[:,:] = 1./ np.sqrt(W[:,:])
        except FloatingPointError:
            msg.bold('\nError!')
            print('Tried to take the square root of a negative Lorentz factor! \nTry checking your velocities?\n')
            sys.exit()

        return W




    def calcu0(self):
        """
        Calculates the timelike coordinate of the 4-velocity using the Lorentz
        factor and alpha, so W = alpha * u0

        Returns
        -------
        u0 : float array
            u0 on grid
        """

        W = self.calcW()

        return W[:,:] / self.alpha[np.newaxis,:]




    def g(self, x):
        """
        Calculates the 2+1-metric at the coordinate x.
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

        met = np.diag([-1., 1., 1.]) #flat default
        met[0,0] = -self.alpha[x[2]]**2 + np.dot(self.beta, self.beta)
        met[0,1:] = np.transpose(self.beta)
        met[1:,0] = self.beta
        met[1:,1:] = self.gamma

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

        christls = np.zeros((3,3,3))

        r = self.cc_data.grid.y[x[2]]
        g = (self.alpha[x[2]]**2 - 1.) * r**2 * 0.5

        #For simple time-lagged metric, only have 3 non-zero christoffels.
        christls[0,0,2] = -g/(self.alpha[x[2]]**2 * r**2)
        christls[0,2,0] = -g/(self.alpha[x[2]]**2 * r**2)
        christls[2,0,0] = -g/r**2


        # For non-simple, we have to do more icky stuff including time and space
        # derivatives of stuff, so I shall not do this for now.

        return christls
