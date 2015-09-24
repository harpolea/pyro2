#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, solver_name, problem_name, outfile, W, H, n=0, vmins=[None, None, None, None], vmaxes=[None, None, None, None]):

    exec ('import ' + solver_name + ' as solver')

    sim = solver.Simulation(solver_name, problem_name, None)
    sim.cc_data = myd
    sim.n = n

    plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')

    sim.dovis(vmins=vmins, vmaxes=vmaxes)
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
    print (usage)
    sys.exit()


if __name__== "__main__":

    #reload(sys)
    #sys.setdefaultencoding('utf-8')

    for i in range(0, 600):
        outfile = "../../Work/pyro/results/rt/rt_256_" +  format(i, '04') + ".png"
        #outfile = "../../Work/pyro/results/kh_1024_" +  format(i, '04') + ".png"
        my_dpi = 96.
        W = 1920/my_dpi
        H = 1080/my_dpi

        # bubble max and mins
        #vmins = [90., 0., -0.00075, -0.2]
        #vmaxes = [105., 0.0021, 0.0021, 0.2]

        # double bubble max and mins
        #vmins = [50., 0., -0.0002, -0.05]
        #vmaxes = [105., 0.0003, 0.0003, 0.05]

        try:
            solver = 'lm_gr'
            #problem = 'bubble'
            #problem = 'kh'
            problem = 'rt'
        except:
            usage()

        try:
            file = "../../Work/pyro/results/rt/rt_256_" +  format(i, '04') + ".pyro"
            #file = "../../Work/pyro/results/kh_1024_" +  format(i, '04') + ".pyro"
        except:
            usage()

        myg, myd = patch.read(file)

        makeplot(myd, solver, problem, outfile, W, H, n=i)#, vmins=vmins, vmaxes=vmaxes)
