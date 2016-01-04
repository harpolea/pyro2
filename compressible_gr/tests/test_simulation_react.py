import eos_defns
import numpy
from numpy.testing import assert_allclose
import simulation
import simulation_react
from unsplitFluxes import *

def test_simulation_react():

    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params("compressible_gr/_defaults")
    rp.load_params("compressible_gr/problems/_test.defaults")
    rp.load_params("compressible_gr/problems/inputs.test")

    # problem-specific runtime parameters
    rp.load_params(solver_name + "/problems/_" +
    sim = simulation.Simulation("compressible_gr", "test", rp)

    my_data = sim.cc_data

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    myg = my_data.grid

    # get conserved and primitive variables.
    D = sim.cc_data.get_var("D")
    Sx = sim.cc_data.get_var("Sx")
    Sy = sim.cc_data.get_var("Sy")
    tau = sim.cc_data.get_var("tau")
    DX = sim.cc_data.get_var("DX")

    gamma = sim.rp.get_param("eos.gamma")
    c = sim.rp.get_param("eos.c")
    _rho = np.zeros_like(D.d)
    _u = np.zeros_like(D.d)
    _v = np.zeros_like(D.d)
    _p = np.zeros_like(D.d)
    _X = np.zeros_like(D.d)

    # we need to compute the primitive speeds and sound speed
    for i in range(myg.qx):
        for j in range(myg.qy):
            U = (D.d[i,j], Sx.d[i,j], Sy.d[i,j], tau.d[i,j], DX.d[i,j])
            names = ['D', 'Sx', 'Sy', 'tau', 'DX']
            nan_check(U, names)
            V, _ = cons_to_prim(U, c, gamma)

            _rho[i,j], _u[i,j], _v[i,j], _, _p[i,j], _X[i,j] = V

    rho = myg.scratch_array()
    u = myg.scratch_array()
    v = myg.scratch_array()
    p = myg.scratch_array()
    X = myg.scratch_array()
    rho.d[:,:] = _rho
    u.d[:,:] = _u
    v.d[:,:] = _v
    p.d[:,:] = _p
    X.d[:,:] = _X

    # test the temperature
    T = sim.calc_T(p, D, X, rho)

    # assert_allclose()

    # test Q and omega_dot

    Q, omega_dot = sim.calc_Q_omega_dot(D, X, rho, T)

    # assert_allclose()

    # test burning fluxes

    (blank, Sx_F, Sy_F, tau_F, DX_F) = sim.burning_flux()

    # assert_allclose()
