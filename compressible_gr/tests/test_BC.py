"""
Test the compressible-specific boundary conditions.
"""

import numpy as np
from numpy.testing import assert_allclose
import BC
import simulation
from util import runparams, msg

def test_user():
    rp = runparams.RuntimeParameters()
    sim = simulation.Simulation("compressible_gr", "test", rp)

    my_data = sim.cc_data

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    myg = my_data.grid

    
