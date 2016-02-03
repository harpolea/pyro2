from numpy.testing import assert_allclose
from compressible_gr.simulation_react import SimulationReact
from util import runparams
import compressible_gr.cons_to_prim as cy
from copy import deepcopy

def test_simulation_react_static():
    """
    Test to check that an initially static system stays that way.
    """

    print('\nRunning test_simulation_react_static')

    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params("compressible_gr/_defaults")
    rp.load_params("compressible_gr/tests/_test.defaults")
    rp.load_params("compressible_gr/tests/inputs.test")

    # problem-specific runtime parameters
    sim = SimulationReact("compressible_gr", "test", rp, testing=True)
    sim.initialize()

    # make a copy to make sure it hasn't changed.
    sim_old = deepcopy(sim)

    init_tstep_factor = rp.get_param("driver.init_tstep_factor")
    max_dt_change = rp.get_param("driver.max_dt_change")
    fix_dt = rp.get_param("driver.fix_dt")
    dt_old = fix_dt

    # get the timestep
    if fix_dt > 0.0:
        sim.dt = fix_dt
        #sim_py.dt = fix_dt
    else:
        sim.compute_timestep()
        if sim.n == 0:
            sim.dt = init_tstep_factor*sim.dt
            #sim_py.dt = init_tstep_factor*sim.dt
        else:
            sim.dt = min(max_dt_change*dt_old, sim.dt)
            #sim_py.dt = min(max_dt_change*dt_old, sim.dt)
        dt_old = sim.dt

    sim.preevolve()
    sim.cc_data.t = 0.0

    # do a few iterations
    nIts = 50
    for i in range(nIts):
        sim.evolve()

    sim.finalize()

    #--------------------------------------------------------------------
    # Check to see if it's changed
    #--------------------------------------------------------------------

    my_data = sim.cc_data

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

    ###########################
    # repeat for original data.
    ###########################

    # get conserved and primitive variables.
    D_old = sim_old.cc_data.get_var("D")
    Sx_old = sim_old.cc_data.get_var("Sx")
    Sy_old = sim_old.cc_data.get_var("Sy")
    tau_old = sim_old.cc_data.get_var("tau")
    DX_old = sim_old.cc_data.get_var("DX")

    rho_old = myg.scratch_array()
    u_old = myg.scratch_array()
    v_old = myg.scratch_array()
    p_old = myg.scratch_array()
    X_old = myg.scratch_array()

    U = myg.scratch_array(sim.vars.nvar)
    U.d[:,:,sim.vars.iD] = D_old.d
    U.d[:,:,sim.vars.iSx] = Sx_old.d
    U.d[:,:,sim.vars.iSy] = Sy_old.d
    U.d[:,:,sim.vars.itau] = tau_old.d
    U.d[:,:,sim.vars.iDX] = DX_old.d

    V = myg.scratch_array(sim.vars.nvar)

    V.d[:,:,:] = cy.cons_to_prim(U.d, c, gamma, myg.qx,
        myg.qy, sim.vars.nvar, sim.vars.iD, sim.vars.iSx,
        sim.vars.iSy, sim.vars.itau, sim.vars.iDX)
    rho_old.d[:,:] = V.d[:,:,sim.vars.irho]
    u_old.d[:,:] = V.d[:,:,sim.vars.iu]
    v_old.d[:,:] = V.d[:,:,sim.vars.iv]
    p_old.d[:,:] = V.d[:,:,sim.vars.ip]
    X_old.d[:,:] = V.d[:,:,sim.vars.iX]

    # finally compare original data to evolved
    assert_allclose([rho.d, u.d, v.d, p.d, X.d],
        [rho_old.d, u_old.d, v_old.d, p_old.d, X_old.d],
        rtol=1.e-10)

    # test the temperature
    T = sim.calc_T(p, D, X, rho)

    # assert_allclose()

    # test Q and omega_dot

    Q, omega_dot = sim.calc_Q_omega_dot(D, X, rho, T)

    # assert_allclose()

    # test burning fluxes

    (blank, Sx_F, Sy_F, tau_F, DX_F) = sim.burning_flux()

    # assert_allclose()
