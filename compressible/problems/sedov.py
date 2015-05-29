from __future__ import print_function

import sys
import mesh.patch as patch
import numpy
from util import msg
import math
import pylsmlib

def init_data(my_data, rp):
    """ initialize the sedov problem """

    msg.bold("initializing the sedov problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")
    phi  = my_data.get_var("phi")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:,:] = 1.0
    xmom[:,:] = 0.0
    ymom[:,:] = 0.0
    phi[:,:]  = -1.0

    E_sedov = 1.0

    r_init = rp.get_param("sedov.r_init")

    gamma = rp.get_param("eos.gamma")
    pi = math.pi

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)


    # initialize the pressure by putting the explosion energy into a
    # volume of constant pressure.  Then compute the energy in a zone
    # from this.
    nsub = 4

    phii = -1.0

    i = my_data.grid.ilo
    while i <= my_data.grid.ihi:

        j = my_data.grid.jlo
        while j <= my_data.grid.jhi:

            dist = numpy.sqrt((my_data.grid.x[i] - xctr)**2 +
                              (my_data.grid.y[j] - yctr)**2)

            if (dist < 2.0*r_init):
                pzone = 0.0

                ii = 0
                while ii < nsub:

                    jj = 0
                    while jj < nsub:

                        xsub = my_data.grid.xl[i] + (my_data.grid.dx/nsub)*(ii + 0.5)
                        ysub = my_data.grid.yl[j] + (my_data.grid.dy/nsub)*(jj + 0.5)

                        dist = numpy.sqrt((xsub - xctr)**2 + \
                                          (ysub - yctr)**2)

                        if dist <= r_init:
                            p = (gamma - 1.0)*E_sedov/(pi*r_init*r_init)
                            phii = 1.0
                        else:
                            p = 1.e-5
                            phii = -1.0

                        pzone += p

                        jj += 1
                    ii += 1

                p = pzone/(nsub*nsub)
            else:
                p = 1.e-5

            ener[i,j] = p/(gamma - 1.0)
            phi[i,j] = phii

            j += 1
        i += 1

        phi[:,:] = pylsmlib.computeDistanceFunction(phi, dx=my_data.grid.dx, order=1)



def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sedov_compare.py can be used to analyze these
          results.  That will perform an average at constant radius and
          compare the radial profiles to the exact solution.  Sample exact
          data is provided as analysis/cylindrical-sedov.out
          """

    print(msg)
