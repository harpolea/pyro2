from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from lm_atm.problems import *
import lm_atm.LM_atm_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
import multigrid.variable_coeff_MG as vcMG
from util import profile

class Metric:

    def __init__(self, cellData, alpha, beta, gamma):
        """
        Initialize the Metric object.

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
        self.W = 1.; #Lorentz factor.
        self.gamma = gamma

    def dets(self):
        """
        Calculates the square roots of the 3- and 4-metric determinants and
        returns them.
        """

        #do stuff

    def g(self, x):
        """
        Calculates the 4-metric at the coordinate x.
        Currently alpha, beta and gamma have no x-dependence so this is kind of
        redundant.

        Parameters
        ----------
        x : float array
            4-coordinate
        """

        met = diag([-1., 1., 1., 1.])
        g[0,0] = -self.alpha**2 + np.dot(self.beta, self.beta)
        g[0,1:] = self.beta'
        g[1:,0] = self.beta
        g[1:,1:] = self.gamma
