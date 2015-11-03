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

    if (n % 10 == 0):
        print("printing: {}".format(n))


def usage():
    usage="""
usage: replot.py [-h] [-o image.png] [-s] solver problem basedir resolution start end

positional arguments:
  solver        required inputs: solver name
  problem       required inputs: problem name
  basedir       required inputs: base directory to read from
  resolution    required inputs: resolution of run
  start         required inputs: first output to render
  end           required inputs: last output to render

optional arguments:
  -h, --help    show this help message and exit
  -o image.png  output image name. The extension .png will generate a PNG
                file, .eps will generate an EPS file (default: plot.png).
  -W width      width in inches
  -H height     height in inches
  -s step       number of steps between renders
"""
    print (usage)
    sys.exit()


if __name__== "__main__":

    #reload(sys)
    #sys.setdefaultencoding('utf-8')

    try:
        opts, next = getopt.getopt(sys.argv[1:], "h:W:H:s:")
    except getopt.GetoptError:
        sys.exit("invalid calling sequence")

    my_dpi = 96.
    W = 1920/my_dpi
    H = 1080/my_dpi
    step = 1

    for o, a in opts:
        if o == "-h":
            usage()
        if o == "-W":
            W = float(a)
        if o == "-H":
            H = float(a)
        if o == "-s":
            step = int(a)

    try:
        solver = next[0]
    except:
        usage()
    try:
        problem = next[1]
    except:
        usage()
    try:
        basedir = next[2]
    except:
        usage()
    try:
        resolution = next[3]
    except:
        usage()
    try:
        start = int(next[4])
    except:
        usage()
    try:
        end = int(next[5])
    except:
        usage()

    for i in range(start, end+1, step):
        base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '04')
        outfile = base + ".png"
        #outfile = "../../Work/pyro/results/kh_1024_" +  format(i, '04') + ".png"

        # bubble max and mins
        #vmins = [90., 0., -0.00075, -0.2]
        #vmaxes = [105., 0.0021, 0.0021, 0.2]

        # double bubble max and mins
        #vmins = [50., 0., -0.0002, -0.05]
        #vmaxes = [105., 0.0003, 0.0003, 0.05]

        try:
            file = base + ".pyro"
            #file = "../../Work/pyro/results/kh_1024_" +  format(i, '04') + ".pyro"
        except:
            usage()

        myg, myd = patch.read(file)

        makeplot(myd, solver, problem, outfile, W, H, n=i)#, vmins=vmins, vmaxes=vmaxes)
