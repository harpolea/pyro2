from util import runparams
import lm_sr.simulation as sn


class TestSimulation(object):
    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """
        pass

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """
        pass

    def setup_method(self):
        """ this is run before each test """
        self.rp = runparams.RuntimeParameters()

        self.rp.params["mesh.nx"] = 8
        self.rp.params["mesh.ny"] = 8

        self.rp.params["mesh.xmin"] = 0
        self.rp.params["mesh.xmax"] = 1
        self.rp.params["mesh.ymin"] = 0
        self.rp.params["mesh.ymax"] = 1

        self.rp.params["mesh.xlboundary"] = "periodic"
        self.rp.params["mesh.xrboundary"] = "periodic"
        self.rp.params["mesh.ylboundary"] = "periodic"
        self.rp.params["mesh.yrboundary"] = "periodic"

        self.rp.params["eos.gamma"] = 1.4
        self.rp.params["lm-atmosphere.grav"] = 1.0

        self.sim = sn.Simulation("lm_sr", "test", self.rp)
        self.sim.initialize()

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_initializationst(self):
        dens = self.sim.cc_data.get_var("density")
        assert dens.min() == 1.0 and dens.max() == 1.0

        xvel = self.sim.cc_data.get_var("x-velocity")
        assert xvel.min() == 0.0 and xvel.max() == 0.0

        yvel = self.sim.cc_data.get_var("y-velocity")
        assert yvel.min() == 0.0 and yvel.max() == 0.0

        eint = self.sim.cc_data.get_var("eint")
        assert eint.min() == 2.5 and eint.max() == 2.5

        rho0 = self.sim.base["rho0"].d
        assert rho0.min() == 1.0 and rho0.max() == 1.0

        p0 = self.sim.base["p0"].d
        assert p0.min() == 1.0 and p0.max() == 1.0
