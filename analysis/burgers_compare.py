#!/usr/bin/env python3

from __future__ import print_function

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from util import msg, runparams, io

usage = """
      compare the output for a burgers problem with the exact solution.

      usage: ./burgers_compare.py file
"""


def abort(string):
    print(string)
    sys.exit(2)


if not len(sys.argv) == 2:
    print(usage)
    sys.exit(2)

try:
    file1 = sys.argv[1]
except IndexError:
    print(usage)
    sys.exit(2)

sim = io.read(file1)
myd = sim.cc_data
myg = myd.grid

# time of file
t = myd.t
if myg.nx > myg.ny:
    # x-problem
    xmin = myg.xmin
    xmax = myg.xmax
    param_file = "inputs.sine"
else:
    # y-problem
    xmin = myg.ymin
    xmax = myg.ymax
    param_file = "inputs.sine"


u = myd.get_var("xvel")
v = myd.get_var("yvel")

# get the 1-d profile from the simulation data -- assume that whichever
# coordinate is the longer one is the direction of the problem

# parameter defaults
rp = runparams.RuntimeParameters()
rp.load_params("../_defaults")
rp.load_params("../burgers/_defaults")
rp.load_params("../burgers/problems/_sine.defaults")

# now read in the inputs file
if not os.path.isfile(param_file):
    # check if the param file lives in the solver's problems directory
    param_file = "../burgers/problems/" + param_file
    if not os.path.isfile(param_file):
        msg.fail("ERROR: inputs file does not exist")

rp.load_params(param_file, no_new=1)

x = myg.x[myg.ilo:myg.ihi + 1]
jj = myg.ny // 2

u = u[myg.ilo:myg.ihi + 1, jj]
ut = v[myg.ilo:myg.ihi + 1, jj]

# print(myg)

x_exact = x
u_exact = np.ones_like(x)

t = 0.1

nu = 1
a = 2

u_exact[:] = x / (1 + t)

# plot
fig, ax = plt.subplots(nrows=1, ncols=1, num=1)

plt.rc("font", size=10)

ax.plot(x_exact, u_exact, label='exact')
ax.scatter(x, u, marker="x", s=7, color="r", label='pyro')

ax.set_ylabel(r"$u$")
ax.set_xlim(0, 1.0)

ax.set_xlabel(r"x")

lgd = ax.legend()

plt.subplots_adjust(hspace=0.25)

fig.set_size_inches(8.0, 8.0)

plt.savefig("burgers_compare.png", bbox_inches="tight")
