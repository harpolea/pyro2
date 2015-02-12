from __future__ import print_function

import numpy as np

from lm_atm.problems import *
import lm_atm.LM_atm_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
import multigrid.variable_coeff_MG as vcMG
from util import profile

class Metric:

    def __init__(self, cellData, alpha, beta, gamma):
        """
        Initialize the Metric object. This is a standard 3+1 metric.

        Parameters
        ----------
        cellData : CellCenterData2d object
            Simulation data object
        alpha : float
            lapse function
        beta : float array
            shift vector
        gamma : float array
            spatial 3-metric
        """


        self.cc_data = cellData
        self.alpha = alpha
        self.beta = beta
        self.W = 1. #Lorentz factor.
        self.gamma = gamma

    def dets(self):
        """
        Calculates the square roots of the 3- and 4-metric determinants and
        returns them.
        """

        sg = self.cc_data.grid.scratch_array()
        sgamma = self.cc_data.grid.scratch_array()

        # FIXME: implement this

        return sg, sgamma

    def calcW(self):
        """
        Calculates the Lorentz factor and returns it.
        """

        W = self.cc_data.grid.scratch_array()

        # FIXME: work out how to calculate this

        return W

    def calcu0(self):
        """
        Calculates the timelike coordinate of the 4-velocity using the Lorentz
        factor and alpha, so W = alpha * u0
        """

        W = self.calcW
        alpha = self.alpha

        return W / alpha



    def g(self, x):
        """
        Calculates the 2+1-metric at the coordinate x.
        Currently alpha, beta and gamma have no x-dependence so this is kind of
        redundant.

        Parameters
        ----------
        x : float array
            2+1-coordinate of point where g is to be calculated
        """

        met = np.diag([-1., 1., 1.])
        met[0,0] = -self.alpha**2 + np.dot(self.beta, self.beta)
        met[0,1:] = np.self.beta.T
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
        """

        christls = np.zeros((3,3,3))

        #K = np.zeros((2,2)) #placeholder

        r = x[2] * self.cc_data.grid.dy
        g = (self.alpha**2 - 1.) / (2. * r)

        #For simple time-lagged metric, only have 3 non-zero christoffels.
        christls[0,0,2] = g/self.alpha**2
        christls[0,2,0] = g/self.alpha**2
        christls[2,0,0] = g


        # For non-simple, we have to do more icky stuff including time and space
        # derivatives of stuff, so I shall not do this for now.

        return christls
