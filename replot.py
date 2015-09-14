#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, solver_name, outfile, W, H):

    exec 'import ' + solver_name + ' as solver'

    sim = solver.Simulation(solver_name, None, None)
    sim.cc_data = myd

    plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')

    sim.dovis()
    plt.savefig(outfile)
    #plt.show()


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

    for i in range(1, 501):
        outfile = "../../Work/pyro/results/bubble_256_" +  format(i, '04') + ".png"
        my_dpi = 96.
        W = 1280/my_dpi
        H = 720/my_dpi

        try:
            solver = 'lm_gr'
        except:
            usage()

        try:
            file = "../../Work/pyro/results/bubble_256_" +  format(i, '04') + ".pyro"
        except:
            usage()

        myg, myd = patch.read(file)

        makeplot(myd, solver, outfile, W, H)
