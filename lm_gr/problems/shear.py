"""
Initialize the doubly periodic shear layer (see, for example, Martin
and Colella, 2000, JCP, 163, 271).  This is run in a unit square
domain, with periodic boundary conditions on all sides.  Here, the
initial velocity is

              / tanh(rho_s (y-0.25))   if y <= 0.5
u(x,y,t=0) = <
              \ tanh(rho_s (0.75-y))   if y > 0.5


v(x,y,t=0) = delta_s sin(2 pi x)


"""

from __future__ import print_function

import numpy as np
import mesh.patch as patch
from util import msg


def init_data(my_data, base_data, rp, metric):
    """
    initialize the incompressible shear problem

    Parameters
    ----------
    my_data : CellCenterMG2d object
        simulation data
    base_data : CellCenterMG1d object
        simulation base states
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    metric: Metric object
        metric for simulation
    """

    msg.bold("initializing the incompressible shear problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in shear.py")

    # get the necessary runtime parameters
    rho_s = rp.get_param("shear.rho_s")
    delta_s = rp.get_param("shear.delta_s")
    dens_base = rp.get_param("shear.dens_base")
    gamma = rp.get_param("eos.gamma")
    c = rp.get_param("lm-gr.c")
    R = rp.get_param("lm-gr.radius")
    grav = rp.get_param("lm-gr.grav")

    # get the velocities
    # get the density and velocities
    dens = my_data.get_var("density")
    enth = my_data.get_var("enthalpy")
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    myg = my_data.grid
    dens.d[:, :] = dens_base
    pres = myg.scratch_array()
    pres.d[:, :] = dens.d**gamma
    eint.d[:, :] = pres.d / ((gamma - 1.0) * dens.d)
    enth.d[:, :] = eint.d + pres.d / dens.d

    y_half = 0.5 * (myg.ymin + myg.ymax)

    print('y_half = ', y_half)
    print('delta_s = ', delta_s)
    print('rho_s = ', rho_s)

    # there is probably an easier way to do this without loops, but
    # for now, we will just do an explicit loop.
    for j in range(myg.jlo, myg.jhi+1):

        if (myg.y[j] <= y_half):
            u.d[:, j] = 1.*np.tanh(rho_s * (myg.y[np.newaxis, j] - 0.25))
            dens.d[:, j] = dens_base * \
                (1. + 0.01 * np.tanh(rho_s * (myg.y[np.newaxis, j] - 0.25)))
        else:
            u.d[:, j] = 1.*np.tanh(rho_s * (0.75 - myg.y[np.newaxis, j]))
            dens.d[:, j] = dens_base * \
                (1. + 0.01 * np.tanh(rho_s * (0.75 - myg.y[np.newaxis, j])))

    v.d[:, myg.jlo: myg.jhi+1] = delta_s * \
        np.sin(2.0 * np.pi * myg.x[:, np.newaxis] + 0.5 * np.pi)

    print("extrema: ", np.min(u.d.flat), np.max(u.d.flat))

    u0 = metric.calcu0()
    u0flat = np.mean(u0.d, axis=0)

    # do the base state by laterally averaging
    D0 = base_data.get_var("D0")
    Dh0 = base_data.get_var("Dh0")

    D0.d[:] = np.mean(dens.d * u0.d, axis=0)
    Dh0.d[:] = np.mean(enth.d * u0.d * dens.d, axis=0)

    # base pressure
    p0 = base_data.get_var("p0")
    p0.d[:] = (D0.d / u0flat[:])**gamma
    base_data.fill_BC("p0")

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - myg.dy * grav * Dh0.d[i] / \
            (u0flat[i] * c**2 * metric.alpha.d[i]**2 * R)

    if (myg.xmin != 0 or myg.xmax != 1 or
            myg.ymin != 0 or myg.ymax != 1):
        msg.fail("ERROR: domain should be a unit square")

    # multiply by correct u0s
    dens.d[:, :] *= u0.d  # rho * u0
    enth.d[:, :] *= u0.d * dens.d  # rho * h * u0

    # fill ghost cells
    my_data.fill_BC("x-velocity")
    my_data.fill_BC("y-velocity")
    my_data.fill_BC("density")
    my_data.fill_BC("enthalpy")
    my_data.fill_BC("eint")
    base_data.fill_BC("D0")
    base_data.fill_BC("Dh0")
    base_data.fill_BC("p0")


def finalize():
    """ print out any information to the user at the end of the run """
    pass
