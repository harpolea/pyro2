import mapped.mapped_grid as mapped
# import mesh.patch as patch
from util import runparams
import numpy as np
from numpy.testing import assert_array_equal
from simulation_null import grid_setup


def test_rectilinear():
    """
    Test a rectilinear grid with dx=dy
    """

    rp = runparams.RuntimeParameters()

    rp.params["mesh.nx"] = 8
    rp.params["mesh.ny"] = 8
    rp.params["mesh.xmin"] = 0
    rp.params["mesh.xmax"] = 1
    rp.params["mesh.ymin"] = 0
    rp.params["mesh.ymax"] = 1
    rp.params["particles.do_particles"] = 0

    # set up grid
    cart_grid = grid_setup(rp, ng=4)
    myg = mapped.Rectilinear(
        cart_grid, hxmin=0, hxmax=1, hymin=0, hymax=1)

    cell_areas = myg.cell_areas

    assert_array_equal(np.ones_like(cell_areas) *
                       myg.cart.dx * myg.cart.dy, cell_areas)

    nx, ny = myg.normals(0)

    assert_array_equal(nx, np.ones_like(nx))
    assert_array_equal(ny, np.zeros_like(ny))

    assert_array_equal(myg.hx2d, myg.cart.x2d)
    assert_array_equal(myg.hy2d, myg.cart.y2d)
