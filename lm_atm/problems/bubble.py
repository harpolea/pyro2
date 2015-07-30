from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg


def init_data(my_data, base_data, rp, metric):
    """
    initialize the bubble problem

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

    msg.bold("initializing the bubble problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    if not isinstance(base_data, patch.CellCenterData1d):
        print("ERROR: patch invalid in bubble.py")
        print(base_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    enth = my_data.get_var("enthalpy")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    grav = rp.get_param("lm-atmosphere.grav")

    gamma = rp.get_param("eos.gamma")
    c = rp.get_param("lm-atmosphere.c")
    R = rp.get_param("lm-atmosphere.radius")

    # scale_height = rp.get_param("bubble.scale_height")
    dens_base = rp.get_param("bubble.dens_base")
    dens_cutoff = rp.get_param("bubble.dens_cutoff")

    x_pert = rp.get_param("bubble.x_pert")
    y_pert = rp.get_param("bubble.y_pert")
    r_pert = rp.get_param("bubble.r_pert")
    pert_amplitude_factor = rp.get_param("bubble.pert_amplitude_factor")

    # initialize the components -- we'll get a pressure too
    # but that is used only to initialize the base state
    xvel.d[:, :] = 0.0
    yvel.d[:, :] = 0.0
    dens.d[:, :] = dens_cutoff
    u0 = metric.calcu0()
    u0flat = np.mean(u0.d, axis=0)

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    # dens[:, myg.jlo:myg.jhi+1] = np.maximum(dens_base * \
    #    np.exp(-myg.y[myg.jlo:myg.jhi+1] / scale_height), dens_cutoff)
    dens.v()[:, :] = dens_base * \
        np.exp(-grav * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
               (gamma * c**2 * R * metric.alpha.v2d()**2))
    # dens[:,myg.jlo:myg.jhi+1] = (dens_base**gamma - grav * \
    #    myg.y[np.newaxis, myg.jlo:myg.jhi+1] / (R * \
    #    metric.alpha[np.newaxis,myg.jlo:myg.jhi+1]**2))**(1./gamma)

    # set the pressure (P = cs2*dens)
    pres.d[:, :] = (dens.d)**gamma
    eint.d[:, :] = pres.d / ((gamma - 1.0) * dens.d)
    enth.d[:, :] = eint.d + pres.d / dens.d  # Newtonian

    # do the base state by laterally averaging
    D0 = base_data.get_var("D0")
    Dh0 = base_data.get_var("Dh0")

    D0.d[:] = np.mean(dens.d * u0.d, axis=0)
    Dh0.d[:] = np.mean(enth.d * u0.d * dens.d, axis=0)

    # boost the specific internal energy, keeping the pressure
    # constant, by dropping the density
    r = np.sqrt((myg.x2d - x_pert)**2 + (myg.y2d - y_pert)**2)

    idx = r <= r_pert
    eint.d[idx] *= pert_amplitude_factor
    dens.d[idx] = pres.d[idx]/(eint.d[idx]*(gamma - 1.0))
    enth.d[idx] = eint.d[idx] + pres.d[idx]/dens.d[idx]

    # base pressure
    p0 = base_data.get_var("p0")
    p0.d[:] = (D0.d / u0flat[:])**gamma

    base_data.fill_BC("p0")

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - myg.dy * grav * Dh0.d[i] / \
            (u0flat[i] * c**2 * metric.alpha.d[i]**2 * R)

    print('pressure: ', pres.d[20, 10:20])
    print('p0: ', p0.d[10:20])

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


def checkXSymmetry(grid, nx):
    """
    Checks to see if a grid is symmetric in the x-direction.

    Parameters
    ----------
    grid : float array
        2d grid to be checked
    nx :
        grid x-dimension

    Returns
    -------
    sym : boolean
        whether or not the grid is symmetric
    """

    halfGrid = grid[-np.floor(nx/2):, :] - grid[np.floor(nx/2)-1::-1, :]
    # sym = True

    if np.max(np.abs(halfGrid)) > 1.e-15:
        print('\nOh no! An asymmetry has occured!\n')
        print('Asymmetry has amplitude: ', np.max(np.abs(halfGrid)))
        # sym = False

    # return sym


def finalize():
    """ print out any information to the user at the end of the run """
    pass
