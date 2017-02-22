#!/usr/bin/env python
"""
Make resulting png plot output into a gif with:
    convert -delay 20 -loop 0 ../../Work/pyro/results/bubble*.png  lm_gr/results/bubble_128.gif

Make into mpeg:
    ffmpeg -framerate 10 -i bubble_256_%04d.png -c:v libx264 -r 10 bubble_256.mp4

If have output e.g. every 5 steps, then use
    ffmpeg -framerate 10 -pattern_type glob -i 'bubble_512_?????.png' -c:v libx264 -r 10 bubble_512.mp4

Note: firefox doesn't like the H264 format, so instead use webm:
    ffmpeg -framerate 10 -pattern_type glob -i 'bubble_512_?????.png' -c:v libvpx -r 10 bubble_512.webm
"""

#import numpy as np
import matplotlib # these two lines set the display so it works
matplotlib.use('Agg') # when ssh into desktop
import matplotlib.pyplot as plt
import sys
import os
import getopt
import importlib
import numpy as np

import mesh.patch as patch
from util import runparams, msg

# plot an output file using the solver's dovis script

def makeplot(myd, solver_name, problem_name, outfile, W, H, n=0, vmins=[None, None, None, None], vmaxes=[None, None, None, None], video=False, magvel_init=None):

    #exec ('import ' + solver_name + ' as solver')
    solver = importlib.import_module(solver_name)

    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params(solver_name + "/_defaults")

    # problem-specific runtime parameters
    rp.load_params(solver_name + "/problems/_" + problem_name + ".defaults")
    param_file = 'inputs.' + problem_name

    # now read in the inputs file
    if not os.path.isfile(param_file):
        # check if the param file lives in the solver's problems directory
        param_file = solver_name + "/problems/" + param_file
        if not os.path.isfile(param_file):
            print(param_file)
            msg.fail("ERROR: inputs file does not exist")

    rp.load_params(param_file, no_new=1)
    init_tstep_factor = rp.get_param("driver.init_tstep_factor")
    max_dt_change = rp.get_param("driver.max_dt_change")
    fix_dt = rp.get_param("driver.fix_dt")

    if solver_name == "lm_gr":
        #if not rp.get_param("lm-gr.cartesian"):
        #    sim = solver.SimulationSpherical(solver_name, problem_name, rp)
        if rp.get_param("lm-gr.react") != 0:
            sim = solver.SimulationReact(solver_name, problem_name, rp)
        else:
            sim = solver.Simulation(solver_name, problem_name, rp)
    elif solver_name == "compressible_gr": # and rp.get_param("compressible-gr.react") != 0:
        sim = solver.SimulationReact(solver_name, problem_name, rp)
    else:
        sim = solver.Simulation(solver_name, problem_name, rp)

    sim.cc_data = myd
    sim.n = n

    if not video:
        plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')
        sim.dovis(vmins=vmins, vmaxes=vmaxes, magvel_init=magvel_init)
        #sim.dovis(vmins=vmins, vmaxes=vmaxes)
    elif solver_name == "compressible_gr":
        plt.figure(num=1, figsize=(0.75*W,1.5*H), dpi=100, facecolor='w')
        sim.dovis_video(vmins=vmins, vmaxes=vmaxes)

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
        opts, next = getopt.getopt(sys.argv[1:], "h:W:H:s:v:")
    except getopt.GetoptError:
        sys.exit("invalid calling sequence")

    my_dpi = 96.
    W = 1920/my_dpi
    H = 1080/my_dpi
    step = 1
    video = False

    for o, a in opts:
        if o == "-h":
            usage()
        if o == "-W":
            W = float(a)
        if o == "-H":
            H = float(a)
        if o == "-s":
            step = int(a)
        if o == "-v":
            video = True

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
        # make it a really high number
        end = int(1e6)
        #usage()

    # compressible_gr
    if solver == "compressible_gr":
        if problem == "kh":
            vmins = [0.001488, 0., 0., -0.5]
            vmaxes = [0.00150, 1.1, 1., 0.1]
        elif problem == "sr_bubble":
            vmins = [0., 0., 0., -6.]
            # M10
            #vmaxes = [0.006, 20., 1., 2.]
            # M5
            vmaxes = [0.005, None, 1., 2.]
        elif problem == 'sod':
            vmins = [0., None, -0.05, 0.]
            vmaxes = [None, None, 1.05, None]
        elif problem == 'swirly':
            vmins = [0.0001, 0., 0., -6.]
            vmaxes = [0.0055, 0.01, 1., 2.]
            if video:
                vmaxes = [0.0055, 1.0]
        else:
            vmins = [None, None, None, None]
            vmaxes = [None, None, None, None]
    elif solver == "lm_gr":
        if problem == "bubble":
            #vmins = [90., 0., 0.0, -0.2]
            #vmaxes = [105., 0.0021, 0.0021, 0.2]
            #vmaxes = [105., 0.0021, 1.0, 0.2]
            vmins = [None, None, None, None]
            vmaxes = [None, None, None, None]
        elif problem == "double_bubble":
            vmins = [50., 0., -0.0002, -0.05]
            vmaxes = [105., 0.0003, 0.0003, 0.05]
            #vmins = [5., 0., 1.7, -0.05]
            #vmaxes = [18., 0.45, 2.6, 1.05]
        elif problem == "ns":
            vmins = [1.e5, 0., None, -1.6e-5]
            vmaxes = [1.015e5, 1., None, 1.6e-5]
        else:
            vmins = [None, None, None, None]
            vmaxes = [None, None, None, None]
    else:
        vmins = [None, None, None, None]
        vmaxes = [None, None, None, None]

    magvel_init = None

    if problem == "gresho":
        ng = 4
        magvel_init = np.zeros((int(resolution)+2*ng, int(resolution)+2*ng))

    for i in range(start, end+1, step):
        if solver == "compressible_gr" and problem == 'kh':
            base = basedir + "/compressible_" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '05')
        else:
            base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '05')
        #outfile = base + "_britgrav.png"
        if video:
            outfile = basedir + "/" + problem + "/" + problem + "_vid_" + str(resolution) + '_' + format(i, '05') + ".png"
        else:
            outfile = base + ".png"

        try:
            file = base + ".pyro"
            #file = "../../Work/pyro/results/kh_1024_" +  format(i, '04') + ".pyro"
        except:
            usage()

        try:
            myg, myd = patch.read(file)
        except:
            try: # backwards compatibility
                if solver == "compressible_gr" and problem == 'kh':
                    base = basedir + "/compressible_" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '04')
                else:
                    base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '04')
                file = base + ".pyro"
                myg, myd = patch.read(file)
            except IOError:
                # file doesn't exist: quietly exit.
                break

        makeplot(myd, solver, problem, outfile, W, H, n=i, vmins=vmins, vmaxes=vmaxes, video=video, magvel_init=magvel_init)
