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

    scale_height = rp.get_param("bubble.scale_height")
    dens_base = rp.get_param("bubble.dens_base")
    dens_cutoff = rp.get_param("bubble.dens_cutoff")

    x_pert = rp.get_param("bubble.x_pert")
    y_pert = rp.get_param("bubble.y_pert")
    r_pert = rp.get_param("bubble.r_pert")
    pert_amplitude_factor = rp.get_param("bubble.pert_amplitude_factor")

    # initialize the components -- we'll get a pressure too
    # but that is used only to initialize the base state
    xvel[:,:] = 0.0
    yvel[:,:] = 0.0
    dens[:,:] = dens_cutoff
    u0 = metric.calcu0()
    u0flat = np.mean(u0[:,:], axis=0)

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    dens[:, myg.jlo:myg.jhi+1] = np.maximum(dens_base * \
        np.exp(-myg.y[myg.jlo:myg.jhi+1] / scale_height), dens_cutoff)


    cs2 = scale_height*abs(grav)

    # set the pressure (P = cs2*dens)
    pres = dens[:,:]**gamma
    eint[:,:] = pres[:,:] /((gamma - 1.0) * dens[:,:])
    enth[:,:] = dens[:,:] + eint[:,:] + pres[:,:]

    # do the base state by laterally averaging
    D0 = base_data.get_var("D0")
    Dh0 = base_data.get_var("Dh0")

    D0[:] = np.mean(dens[:,:], axis=0)
    Dh0[:] = np.mean(enth[:,:], axis=0)

    for i in range(myg.ilo, myg.ihi+1):
        for j in range(myg.jlo, myg.jhi+1):

            r = np.sqrt((myg.x[i] - (x_pert+myg.xmin))**2  \
                + (myg.y[j] - (y_pert+myg.ymin))**2)

            if r <= r_pert:
                # boost the specific internal energy, keeping the pressure
                # constant by dropping the density
                #eint[i,j] *= (1. + (pert_amplitude_factor-1.)*(r_pert-r)/r_pert)
                eint[i,j] *= pert_amplitude_factor
                dens[i,j] = pres[i,j]/(eint[i,j]*(gamma - 1.0))
                enth[i,j] = dens[i,j] + eint[i,j] + pres[i,j]


    p0 = base_data.get_var("p0")

    # redo the pressure via HSE
    #FIXME: need to divide by u0 here???


    p0[:] = (D0[:] + Dh0[:]) * (gamma - 1.) / (u0flat[:] * (2. - gamma))
    p0[1:] = p0[:-1] + 0.5 * myg.dy * (D0[1:] + D0[:-1]) * grav/ \
        (u0flat[1:] * myg.y[1:]**2)

    #p0[myg.jlo+1:myg.jhi+1] = p0[myg.jlo:myg.jhi] + 0.5 * myg.dy * \
    #    (D0[myg.jlo+1:myg.jhi+1] + D0[myg.jlo:myg.jhi]) * \
    #    grav/myg.y[myg.jlo+1:myg.jhi+1]**2


    #fill ghost cells
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
