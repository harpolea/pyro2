#!/usr/bin/env python
"""
Generates plots used in transfer thesis.

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

import mesh.patch as patch
from util import runparams, msg

# plot an output file using the solver's dovis script

def makeplot(myd_i, myd_r, solver_name, problem_name, outfile, W, H, n=0, vmins=[None, None, None, None], vmaxes=[None, None, None, None]):

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
        #if rp.get_param("lm-gr.react") != 0:
        sim = solver.SimulationReact(solver_name, problem_name, rp)
        #else:
        #    sim = solver.Simulation(solver_name, problem_name, rp)

        sim.cc_data = myd_i
        sim.n = n
        if problem_name == 'rt':
            plt.figure(num=1, figsize=(W,0.5*H), dpi=100, facecolor='w')
        elif problem_name == 'kh':
            plt.figure(num=1, figsize=(W,0.3*H), dpi=100, facecolor='w')
        else:
            plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')
        sim.dovis_thesis(vmins=vmins, vmaxes=vmaxes)

    elif solver_name == "compressible_gr" and (problem_name == 'sr_bubble' or problem_name == 'swirly'): # and rp.get_param("compressible-gr.react") != 0:
        sim_i = solver.SimulationReact(solver_name, problem_name, rp)
        sim_r = solver.SimulationReact(solver_name, problem_name, rp)

        sim_i.cc_data = myd_i
        sim_i.n = n
        sim_r.cc_data = myd_r
        sim_r.n = n

        plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')
        sim_i.dovis_thesis(sim_r, vmins=vmins, vmaxes=vmaxes)
    else:
        sim = solver.Simulation(solver_name, problem_name, rp)
        sim.cc_data = myd_i
        sim.n = n
        plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')
        sim.dovis_thesis(vmins=vmins, vmaxes=vmaxes)

    if problem_name == 'sod':
        lgd = plt.legend(bbox_to_anchor=(1.05, 2), loc=2, borderaxespad=0.)
        plt.rc("font", size=18)
        plt.savefig(outfile, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
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
    W = 1280/my_dpi
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
            vmaxes = [0.004, None, 1., 2.]
        elif problem == 'sod':
            #vmins = [0., None, -0.05, 0.]
            vmins = [None, None, None, None]
            vmaxes = [None, None, 1.05, None]
        elif problem == 'swirly':
            #vmins = [0., None, -0.05, 0.]
            vmins = [0.0001, 0., 0., -6.]
            vmaxes = [0.0055, 0.01, 1., 2.]
        else:
            vmins = [None, None, None, None]
            vmaxes = [None, None, None, None]
    elif solver == "lm_gr":
        if problem == "bubble":
            vmins = [90., 0.0, 0.0, 1.28]
            #vmaxes = [105., 0.0021, 0.0021, 0.2]
            vmaxes = [105., 0.02, 1.0, 1.4]
        elif problem == "double_bubble":
            vmins = [50., 0., -0.0002, -0.05]
            vmaxes = [105., 0.0003, 0.0003, 0.05]
            #vmins = [5., 0., 1.7, -0.05]
            #vmaxes = [18., 0.45, 2.6, 1.05]
        elif problem == "ns":
            vmins = [1.e5, 0., None, -1.6e-5]
            vmaxes = [1.015e5, 1., None, 1.6e-5]
        elif problem == 'rt':
            vmins = [99.0, 0.0]
            vmaxes = [101.0, 1.0]
        elif problem == 'kh':
            vmins = [100.0, 0.0]
            vmaxes = [101.1, 1.0]
        else:
            vmins = [None, None, None, None]
            vmaxes = [None, None, None, None]
    else:
        vmins = [None, None, None, None]
        vmaxes = [None, None, None, None]


    for i in range(start, end+1, step):
        if solver == "compressible_gr":
            if problem == 'sod':
                base_i = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_rp2_hlle_' + format(i, '05')
                base_r = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_rp2_' + format(i, '05')
                base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_hlle_' + format(i, '05')
            elif problem == 'sr_bubble' or problem == 'swirly':
                base_i = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_iM5_' + format(i, '05')
                base_r = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_M5_' + format(i, '05')
                base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '05')
            else:
                base_i = basedir + "/compressible_" + problem + "/" + problem + "_" + str(resolution) + '_iM5_' + format(i, '05')
                base_r = basedir + "/compressible_" + problem + "/" + problem + "_" + str(resolution) + '_M5_' + format(i, '05')
                base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '05')
        else:
            base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '05')
        #outfile = base + "_britgrav.png"
        outfile = base + "_transfer.png"

        try:
            if solver == "compressible_gr":
                file_i = base_i + ".pyro"
                file_r = base_r + ".pyro"
                #file = "../../Work/pyro/results/kh_1024_" +  format(i, '04') + ".pyro"
                myg_i, myd_i = patch.read(file_i)
                myg_r, myd_r = patch.read(file_r)
            else:
                file = base + ".pyro"
                myg_i, myd_i = patch.read(file)
                myd_r = None
        except:
            try: # backwards compatibility
                if solver == "compressible_gr":
                    if problem == 'sod':
                        base_i = basedir + "/compressible_" + problem + "/" + problem + "_" + str(resolution) + '_rp2_hlle_' + format(i, '04')
                        base_r = basedir + "/compressible_" + problem + "/" + problem + "_" + str(resolution) + '_rp2_' + format(i, '04')
                        base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_hlle_' + format(i, '04')
                    else:
                        base_i = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_iM5_' + format(i, '04')
                        base_r = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_M5_' + format(i, '04')
                        base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '04')

                    file_i = base_i + ".pyro"
                    file_r = base_r + ".pyro"
                    myg_i, myd_i = patch.read(file_i)
                    myg_r, myd_r = patch.read(file_r)
                else:
                    base = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_' + format(i, '04')
                    file = base + ".pyro"
                    myg_i, myd_i = patch.read(file)
                    myd_r = None
            except IOError:
                # file doesn't exist: quietly exit.
                print('whereami')
                break

        makeplot(myd_i, myd_r, solver, problem, outfile, W, H, n=i, vmins=vmins, vmaxes=vmaxes)