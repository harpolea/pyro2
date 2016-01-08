#!/usr/bin/env python
# python2 plot_diff.py -s 1 compressible_gr sr_bubble ../../Documents/Work/pyro/results 128 0 250

# this doesn't work amazingly, as the simulations were out of sync timewise by about dt = 1.e-4
# could do this better by interpolating to line up the times better, but it would probably be easier to run the two simulations at the same time with exactly the same time steps.

import numpy as np
import matplotlib # these two lines set the display so it works
matplotlib.use('Agg') # when ssh into desktop
import matplotlib.pyplot as plt
import sys
import os
import getopt

import mesh.patch as patch
from util import runparams, msg
from compressible_gr.unsplitFluxes import *


# plot an output file using the solver's dovis script

def makeplot(myd, myd_burn, solver_name, problem_name, outfile, W, H, n=0, vmins=[None, None, None, None], vmaxes=[None, None, None, None]):

    exec ('import ' + solver_name + ' as solver')

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

    sim = solver.SimulationReact(solver_name, problem_name, rp)
    sim_burn = solver.SimulationReact(solver_name, problem_name, rp)

    sim.cc_data = myd
    myg = sim.cc_data.grid
    sim_burn.cc_data = myd_burn
    sim.n = n

    plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')

    plt.clf()
    plt.rc("font", size=12)

    # process sim data
    D = sim.cc_data.get_var("D")
    DX = sim.cc_data.get_var("DX")
    Sx = sim.cc_data.get_var("Sx")
    Sy = sim.cc_data.get_var("Sy")
    tau = sim.cc_data.get_var("tau")

    gamma = sim.cc_data.get_aux("gamma")
    c = sim.cc_data.get_aux("c")
    u = np.zeros_like(D.d)
    v = np.zeros_like(D.d)

    rho = myg.scratch_array()
    p = myg.scratch_array()
    h = myg.scratch_array()
    X = myg.scratch_array()
    _u = myg.scratch_array()

    for i in range(myg.qx):
        for j in range(myg.qy):
            F = (D.d[i,j], Sx.d[i,j], Sy.d[i,j], tau.d[i,j], DX.d[i,j])
            Fp, cs = cons_to_prim(F, c, gamma)
            rho.d[i,j], u[i,j], v[i,j], h.d[i,j], p.d[i,j], X.d[i,j] = Fp

    # get the pressure
    magvel = myg.scratch_array()
    magvel.d[:,:] = np.sqrt(u**2 + v**2)

    T = sim.calc_T(p, D, X, rho)
    T.d[:,:] = np.log(T.d)
    _u.d[:,:] = u

    # now process burnt sim data
    D_burn = sim_burn.cc_data.get_var("D")
    DX_burn = sim_burn.cc_data.get_var("DX")
    Sx_burn = sim_burn.cc_data.get_var("Sx")
    Sy_burn = sim_burn.cc_data.get_var("Sy")
    tau_burn = sim_burn.cc_data.get_var("tau")

    gamma = sim_burn.cc_data.get_aux("gamma")
    c = sim_burn.cc_data.get_aux("c")
    u_burn = np.zeros_like(D.d)
    v_burn = np.zeros_like(D.d)

    rho_burn = myg.scratch_array()
    p_burn = myg.scratch_array()
    h_burn = myg.scratch_array()
    X_burn = myg.scratch_array()
    _u_burn = myg.scratch_array()

    for i in range(myg.qx):
        for j in range(myg.qy):
            F = (D_burn.d[i,j], Sx_burn.d[i,j], Sy_burn.d[i,j], tau_burn.d[i,j], DX_burn.d[i,j])
            Fp, cs = cons_to_prim(F, c, gamma)
            rho_burn.d[i,j], u_burn[i,j], v_burn[i,j], h_burn.d[i,j], p_burn.d[i,j], X_burn.d[i,j] = Fp

    # get the pressure
    magvel_burn = myg.scratch_array()
    magvel_burn.d[:,:] = np.sqrt(u**2 + v**2)

    T_burn = sim_burn.calc_T(p_burn, D_burn, X_burn, rho_burn)
    T_burn.d[:,:] = np.log(T_burn.d)
    _u_burn.d[:,:] = u_burn

    # set sim data to be difference of burnt and unburnt
    for i in range(sim.vars.nvar):
        var = sim.cc_data.get_var_by_index(i)
        var_burn = sim_burn.cc_data.get_var_by_index(i)

        var.d[:,:] -= var_burn.d

    L_x = myg.xmax - myg.xmin
    L_y = myg.ymax - myg.ymin


    fields = [rho, _u, T, X]
    burnt_fields = [rho_burn, _u_burn, T_burn, X_burn]
    field_names = [r"$\rho$", r"$u$", "$\ln(T)$", "$X$"]
    colours = ['blue', 'red', 'black', 'green']

    fig, axes = plt.subplots(nrows=4, ncols=2, num=1)
    orientation = "horizontal"
    if (L_x > 4.*L_y):
        shrink = 0.75

    onLeft = list(range(sim.vars.nvar))

    sparseX = 0
    allYlabel = 1


    for nn in range(4):
        ax = axes.flat[2*nn]
        ax2 = axes.flat[2*nn+1]

        v = fields[nn]
        v_burn = burnt_fields[nn]
        # masking to floating point errors
        v.d[np.fabs(v.d) > 1.e-15] = (v.d[np.fabs(v.d) > 1.e-15] - v_burn.d[np.fabs(v.d) > 1.e-15])/ v.d[np.fabs(v.d) > 1.e-15]
        v.d[np.fabs(v.d) <= 1.e-15] = 0.
        ycntr = np.round(0.5 * myg.qy).astype(int)
        img = ax.imshow(np.transpose(v.v()),
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax], vmin=vmins[nn], vmax=vmaxes[nn])
        plt2 = ax2.plot(myg.x, v.d[:,ycntr], c=colours[nn])
        ax2.set_xlim([myg.xmin, myg.xmax])


        #ax.set_xlabel("x")
        if nn==3:
            ax2.set_xlabel("$x$")
        if nn == 0:
            ax.set_ylabel("$y$")
            ax2.set_ylabel(field_names[nn], rotation='horizontal')
        elif allYlabel:
            ax.set_ylabel("$y$")
            ax2.set_ylabel(field_names[nn], rotation='horizontal')

        ax.set_title(field_names[nn])

        if not nn in onLeft:
            ax.yaxis.offsetText.set_visible(False)
            ax2.yaxis.offsetText.set_visible(False)
            if n > 0:
                ax.get_yaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)

        if sparseX:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax2.xaxis.set_major_locator(plt.MaxNLocator(3))

        ax2.set_ylim([vmins[nn], vmaxes[nn]])
        plt.colorbar(img, ax=ax, orientation=orientation, shrink=0.75)


    plt.figtext(0.05,0.0125, "n: %4d,   t = %10.5f" % (sim.n, sim.cc_data.t))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.4, wspace=0.1)

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
        base_burn = basedir + "/" + problem + "/" + problem + "_hllc_" + str(resolution) + '_' + format(i, '04')
        outfile = basedir + "/" + problem + "/" + problem + "_" + str(resolution) + '_diff_' + format(i, '04') + ".png"
        #outfile = "../../Work/pyro/results/kh_1024_" +  format(i, '04') + ".png"

        # bubble max and mins
        #vmins = [90., 0., -0.00075, -0.2]
        #vmaxes = [105., 0.0021, 0.0021, 0.2]

        # double bubble max and mins
        #vmins = [50., 0., -0.0002, -0.05]
        #vmaxes = [105., 0.0003, 0.0003, 0.05]
        #vmins = [5., 0., 1.7, -0.05]
        #vmaxes = [18., 0.45, 2.6, 1.05]
        vmins=[None, None, None, None]
        vmaxes=[None, None, None, None]

        try:
            file = base + ".pyro"
            file_burn = base_burn + ".pyro"
        except:
            usage()

        myg, myd = patch.read(file)
        myg_burn, myd_burn = patch.read(file_burn)

        makeplot(myd, myd_burn, solver, problem, outfile, W, H, n=i, vmins=vmins, vmaxes=vmaxes)
