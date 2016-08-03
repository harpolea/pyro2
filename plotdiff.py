#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, myd_py, solver_name, problem_name, outfile, W, H, n=0, vmins=[None, None, None, None], vmaxes=[None, None, None, None]):

    exec 'import ' + solver_name + ' as solver'

    sim = solver.Simulation(solver_name, problem_name, None)
    sim.cc_data = myd
    sim_py = solver.Simulation(solver_name, problem_name, None)
    sim_py.cc_data = myd_py
    sim.n = n

    # change sim data to make it the difference
    D = sim.cc_data.get_var("density")
    Dh = sim.cc_data.get_var("enthalpy")
    u = sim.cc_data.get_var("x-velocity")
    v = sim.cc_data.get_var("y-velocity")

    D_py = sim_py.cc_data.get_var("density")
    Dh_py = sim_py.cc_data.get_var("enthalpy")
    u_py = sim_py.cc_data.get_var("x-velocity")
    v_py = sim_py.cc_data.get_var("y-velocity")

    D.d[:,:] = np.log(np.abs(2. * (D.d - D_py.d)/(np.abs(D.d) + np.abs(D_py.d))))
    Dh.d[:,:] = np.log(np.abs(2. * (Dh.d - Dh_py.d)/(np.abs(Dh.d) + np.abs(Dh_py.d))))
    u.d[:,:] = np.log(np.abs(2. * (u.d - u_py.d)/(np.abs(u.d) + np.abs(u_py.d))))
    v.d[:,:] = np.log(np.abs(2. * (v.d - v_py.d)/(np.abs(v.d) + np.abs(v_py.d))))

    plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')

    sim.dovis(vmins=vmins, vmaxes=vmaxes)
    plt.savefig(outfile)


def usage():
    usage="""
usage: plot.py [-h] [-o image.png] solver filename

positional arguments:
  solver        required inputs: solver name
  filename      required inputs: filename to read from

optional arguments:
  -h, --help    show this help message and exit
  -o image.png  output image name. The extension .png will generate a PNG
                file, .eps will generate an EPS file (default: plot.png).
  -W width      width in inches
  -H height     height in inches
"""
    print usage
    sys.exit()


if __name__== "__main__":

    for i in range(51, 151):
        outfile = "../../Work/pyro/results/bubble_128_diff_" +  format(i, '04') + ".png"
        my_dpi = 96.
        W = 1920/my_dpi
        H = 1080/my_dpi

        # bubble max and mins
        #vmins = [89., 0., -0.00075, -0.2]
        #vmaxes = [101., 0.0015, 0.0015, 0.2]

        try:
            solver = 'lm_gr'
            problem = 'bubble'
            #problem = 'kh'
        except:
            usage()

        try:
            file = "../../Work/pyro/results/bubble_128_" +  format(i, '04') + ".pyro"
            file_py = "../../Work/pyro/results/bubble_128_py_" +  format(i, '04') + ".pyro"
        except:
            usage()

        myg, myd = patch.read(file)
        _, myd_py = patch.read(file_py)

        makeplot(myd, myd_py, solver, problem, outfile, W, H, n=i)
