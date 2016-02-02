"""
Test the compressible-specific boundary conditions.
"""

import numpy as np
from numpy.testing import assert_allclose
from compressible_gr.simulation import Simulation
from util import runparams, msg
import compressible_gr.cons_to_prim as cy

def test_user():
    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params("compressible_gr/_defaults")
    rp.load_params("compressible_gr/tests/_test.defaults")
    rp.load_params("compressible_gr/tests/inputs.test")

    sim = Simulation("compressible_gr", "test", rp, testing=True)
    sim.initialize()

    my_data = sim.cc_data

    myg = my_data.grid
    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    # get conserved and primitive variables.
    D = sim.cc_data.get_var("D")
    Sx = sim.cc_data.get_var("Sx")
    Sy = sim.cc_data.get_var("Sy")
    tau = sim.cc_data.get_var("tau")
    DX = sim.cc_data.get_var("DX")

    gamma = sim.rp.get_param("eos.gamma")
    c = sim.rp.get_param("eos.c")
    rho = myg.scratch_array()
    u = myg.scratch_array()
    v = myg.scratch_array()
    p = myg.scratch_array()
    X = myg.scratch_array()

    U = myg.scratch_array(sim.vars.nvar)
    U.d[:,:,sim.vars.iD] = D.d
    U.d[:,:,sim.vars.iSx] = Sx.d
    U.d[:,:,sim.vars.iSy] = Sy.d
    U.d[:,:,sim.vars.itau] = tau.d
    U.d[:,:,sim.vars.iDX] = DX.d

    V = myg.scratch_array(sim.vars.nvar)

    V.d[:,:,:] = cy.cons_to_prim(U.d, c, gamma, myg.qx, myg.qy, sim.vars.nvar, sim.vars.iD, sim.vars.iSx, sim.vars.iSy, sim.vars.itau, sim.vars.iDX)
    rho.d[:,:] = V.d[:,:,sim.vars.irho]
    u.d[:,:] = V.d[:,:,sim.vars.iu]
    v.d[:,:] = V.d[:,:,sim.vars.iv]
    p.d[:,:] = V.d[:,:,sim.vars.ip]
    X.d[:,:] = V.d[:,:,sim.vars.iX]
