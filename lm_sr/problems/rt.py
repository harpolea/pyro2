"""
Implements the Rayleigh-Taylor problem using (mostly) the initial data in the MAESTRO test problem
"""
from __future__ import print_function

import sys
import mesh.patch as patch
import lm_sr.eos as eos
import numpy
from util import msg


def init_data(my_data, base, rp):
    """ initialize the Rayleigh-Taylor problem """

    msg.bold("initializing the Rayleigh-Taylor problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in rt.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    grav = rp.get_param("lm-atmosphere.grav")

    gamma = rp.get_param("eos.gamma")

    rho_1 = rp.get_param("rt.rho_1")
    rho_2 = rp.get_param("rt.rho_2")
    dens_cutoff = rp.get_param("rt.dens_cutoff")
    vel_amplitude = rp.get_param("rt.vel_amplitude")
    vel_width = rp.get_param("rt.vel_width")

    p0_base = rp.get_param("rt.p0_base")

    # initialize the components -- we'll get a pressure too
    # but that is used only to initialize the base state
    xvel[:, :] = 0.0
    yvel[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # as xvel=yvel=0, W=1 and we don't need to update the velocity or the dens to make them the Wilson variables.

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    yctr = 0.5 * (myg.ymin + myg.ymax)

    idxl = myg.y2d <= yctr
    idxh = myg.y2d > yctr

    dens[idxl] = rho_1
    dens[idxh] = rho_2

    pres[:, :] = p0_base

    # do the pressure via HSE
    for j in range(myg.jlo+1, myg.jhi):
        pres[:, j] = pres[:, j-1] + 0.5*myg.dy*(dens[:, j] + dens[:, j-1])*grav

    eint[:, :] = eos.rhoe(gamma, pres) / dens

    # initialise the velocity
    L_x = myg.xmax - myg.xmin
    pert = vel_amplitude * 0.5 * (numpy.cos(2*numpy.pi*myg.x2d / L_x) +
                                  numpy.cos(2*numpy.pi*(L_x-myg.x2d)/L_x))
    pert_height = 0.01 * 0.5 * (numpy.cos(2*numpy.pi*myg.x2d / L_x) +
                                numpy.cos(2*numpy.pi*(L_x-myg.x2d)/L_x)) + 0.5

    yvel[:, :] = numpy.exp(-(myg.y2d - yctr)**2 / vel_width**2) * pert

    dens[:, :] = rho_1 + 0.5 * (rho_2 - rho_1) * \
        (1 + numpy.tanh((myg.y2d-pert_height)/0.005))

    # print(f'tanh = {numpy.max(numpy.tanh((myg.y2d-pert_height)/0.005))}')

    W = 1.0 / numpy.sqrt(1.0 - yvel**2)

    dens[:, :] *= W
    xvel[:, :] /= W
    yvel[:, :] /= W

    # do the base state
    base["rho0"].d[:] = numpy.mean(dens, axis=0)
    base["p0"].d[:] = numpy.mean(pres, axis=0)

    # redo the pressure via HSE
    for j in range(myg.jlo+1, myg.jhi):
        base["p0"].d[j] = base["p0"].d[j-1] + \
            0.5*myg.dy*(base["rho0"].d[j]/numpy.mean(W[:, j]) +
            base["rho0"].d[j-1]/numpy.mean(W[:, j-1]))*grav

    # print(base["p0"].d)
    # print(base["rho0"].d)
    # exit()


def finalize():
    """ print out any information to the user at the end of the run """
    pass
