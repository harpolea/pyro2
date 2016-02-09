from numpy.testing import assert_allclose
from util import runparams
from copy import deepcopy
from lm_gr.simulation import Simulation

def test_simulation_static():
    """
    Test to check that an initially static system stays that way.
    """

    print('\nRunning test_simulation_static')

    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params("lm_gr/_defaults")
    rp.load_params("lm_gr/tests/_test.defaults")
    rp.load_params("lm_gr/tests/inputs.test")

    # problem-specific runtime parameters
    sim = Simulation("lm_gr", "static", rp, testing=True)
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

    # extract data from simulation
    my_data = sim.cc_data
    aux_data = sim.aux_data

    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    myg = my_data.grid

    # get the density, momenta, and energy as separate variables
    rho = my_data.get_var("density")
    h = my_data.get_var("enthalpy")
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = aux_data.get_var("eint")
    scalar = my_data.get_var("scalar")
    T = my_data.get_var("temperature")
    DX = my_data.get_var("mass-frac")

    ###########################
    # repeat for original data.
    ###########################

    old_data = sim_old.cc_data
    old_aux_data = sim_old.aux_data

    # get the density, momenta, and energy as separate variables
    rho_old = old_data.get_var("density")
    h_old = old_data.get_var("enthalpy")
    u_old = old_data.get_var("x-velocity")
    v_old = old_data.get_var("y-velocity")
    eint_old = old_aux_data.get_var("eint")
    scalar_old = old_data.get_var("scalar")
    T_old = old_data.get_var("temperature")
    DX_old = old_data.get_var("mass-frac")

    # finally compare original data to evolved
    assert_allclose([rho.d, u.d, v.d, h.d, eint.d, scalar.d, DX.d], [rho_old.d, u_old.d, v_old.d, h_old.d, eint_old.d, scalar_old.d, DX_old.d], rtol=1.e-8)


def test_simulation_xvel():
    """
    Test to check that system with initial constant velocity in the x-direction evolves as it should.
    """

    print('\nRunning test_simulation_xvel')

    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params("lm_gr/_defaults")
    rp.load_params("lm_gr/tests/_test.defaults")
    rp.load_params("lm_gr/tests/inputs.test")

    # problem-specific runtime parameters
    sim = Simulation("lm_gr", "static", rp, testing=True)
    sim.initialize()

    # set velocity
    u = sim.cc_data.get_var("x-velocity")
    u.d[:,:] = 1.e-8

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

    # extract data from simulation
    my_data = sim.cc_data
    aux_data = sim.aux_data

    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    myg = my_data.grid

    # get the density, momenta, and energy as separate variables
    rho = my_data.get_var("density")
    h = my_data.get_var("enthalpy")
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = aux_data.get_var("eint")
    scalar = my_data.get_var("scalar")
    T = my_data.get_var("temperature")
    DX = my_data.get_var("mass-frac")

    ###########################
    # repeat for original data.
    ###########################

    old_data = sim_old.cc_data
    old_aux_data = sim_old.aux_data

    # get the density, momenta, and energy as separate variables
    rho_old = old_data.get_var("density")
    h_old = old_data.get_var("enthalpy")
    u_old = old_data.get_var("x-velocity")
    v_old = old_data.get_var("y-velocity")
    eint_old = old_aux_data.get_var("eint")
    scalar_old = old_data.get_var("scalar")
    T_old = old_data.get_var("temperature")
    DX_old = old_data.get_var("mass-frac")

    # finally compare original data to evolved
    assert_allclose(
        [rho.v()],
        [rho_old.v()], rtol=1.e-2, atol=1.e-9)
    assert_allclose(
        [h.v(), eint.v()],
        [h_old.v(), eint_old.v()], rtol=1.e-2, atol=1.e-9)
    assert_allclose(
        [scalar.v(), DX.v()],
        [scalar_old.v(), DX_old.v()], rtol=1.e-2, atol=1.e-9)

    assert_allclose(
        [u.v()],
        [u_old.v()], rtol=1.e-2, atol=1.e-9)
    assert_allclose(
        [v.v()],
        [v_old.v()], rtol=1.e-2, atol=1.e-9)


def test_simulation_yvel():
    """
    Test to check that system with initial constant velocity in the y-direction evolves as it should.
    """

    print('\nRunning test_simulation_yvel')

    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params("lm_gr/_defaults")
    rp.load_params("lm_gr/tests/_test.defaults")
    rp.load_params("lm_gr/tests/inputs.test")

    # problem-specific runtime parameters
    sim = Simulation("lm_gr", "static", rp, testing=True)
    sim.initialize()

    # set velocity
    v = sim.cc_data.get_var("y-velocity")
    v.d[:,:] = 1.e-8

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
    nIts = 10
    for i in range(nIts):
        sim.evolve()

    sim.finalize()

    #--------------------------------------------------------------------
    # Check to see if it's changed
    #--------------------------------------------------------------------

    # extract data from simulation
    my_data = sim.cc_data
    aux_data = sim.aux_data

    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    myg = my_data.grid

    # get the density, momenta, and energy as separate variables
    rho = my_data.get_var("density")
    h = my_data.get_var("enthalpy")
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = aux_data.get_var("eint")
    scalar = my_data.get_var("scalar")
    T = my_data.get_var("temperature")
    DX = my_data.get_var("mass-frac")

    ###########################
    # repeat for original data.
    ###########################

    old_data = sim_old.cc_data
    old_aux_data = sim_old.aux_data

    # get the density, momenta, and energy as separate variables
    rho_old = old_data.get_var("density")
    h_old = old_data.get_var("enthalpy")
    u_old = old_data.get_var("x-velocity")
    v_old = old_data.get_var("y-velocity")
    eint_old = old_aux_data.get_var("eint")
    scalar_old = old_data.get_var("scalar")
    T_old = old_data.get_var("temperature")
    DX_old = old_data.get_var("mass-frac")

    # finally compare original data to evolved
    assert_allclose(
        [rho.v()],
        [rho_old.v()], rtol=1.e-2, atol=1.e-9)
    assert_allclose(
        [h.v(), eint.v()],
        [h_old.v(), eint_old.v()], rtol=1.e-2, atol=1.e-9)
    assert_allclose(
        [scalar.v(), DX.v()],
        [scalar_old.v(), DX_old.v()], rtol=1.e-2, atol=1.e-9)

    assert_allclose(
        [u.v()],
        [u_old.v()], rtol=1.e-2, atol=1.e-9)
    assert_allclose(
        [v.v()],
        [v_old.v()], rtol=1.e-2, atol=1.e-9)

# FIXME: add back in
def _test_simulation_hydrostatic():
    """
    Test to check that an initially hydrostatic system stays that way.
    """

    print('\nRunning test_simulation_hydrostatic')

    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params("lm_gr/_defaults")
    rp.load_params("lm_gr/tests/_test.defaults")
    rp.load_params("lm_gr/tests/inputs.hydrostatic")

    # problem-specific runtime parameters
    sim = Simulation("lm_gr", "hydrostatic", rp, testing=True)
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
    nIts = 10
    for i in range(nIts):
        sim.evolve()

    sim.finalize()

    #--------------------------------------------------------------------
    # Check to see if it's changed
    #--------------------------------------------------------------------

    # extract data from simulation
    my_data = sim.cc_data
    aux_data = sim.aux_data

    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    myg = my_data.grid

    # get the density, momenta, and energy as separate variables
    rho = my_data.get_var("density")
    h = my_data.get_var("enthalpy")
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = aux_data.get_var("eint")
    scalar = my_data.get_var("scalar")
    T = my_data.get_var("temperature")
    DX = my_data.get_var("mass-frac")

    ###########################
    # repeat for original data.
    ###########################

    old_data = sim_old.cc_data
    old_aux_data = sim_old.aux_data

    # get the density, momenta, and energy as separate variables
    rho_old = old_data.get_var("density")
    h_old = old_data.get_var("enthalpy")
    u_old = old_data.get_var("x-velocity")
    v_old = old_data.get_var("y-velocity")
    eint_old = old_aux_data.get_var("eint")
    scalar_old = old_data.get_var("scalar")
    T_old = old_data.get_var("temperature")
    DX_old = old_data.get_var("mass-frac")

    # finally compare original data to evolved
    assert_allclose([rho.d, u.d, v.d, h.d, eint.d, scalar.d, DX.d], [rho_old.d, u_old.d, v_old.d, h_old.d, eint_old.d, scalar_old.d, DX_old.d], rtol=1.e-8)
