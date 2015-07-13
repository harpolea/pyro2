#!/usr/bin/env python

import numpy as np
import mesh.patch as patch
import sys
import math

usage = """
      compare the output in file from the incompressible converge problem to
      the analytic solution.

      usage: ./incomp_converge_error.py file
"""

def abort(string):
    print string
    sys.exit(2)


if not len(sys.argv) == 2:
    print usage
    sys.exit(2)


try: file1 = sys.argv[1]
except:
    print usage
    sys.exit(2)

myg, myd = patch.read(file1)


# numerical solution
u = myd.get_var("x-velocity")
v = myd.get_var("y-velocity")

t = myd.t

# analytic solution
u_exact = 1.0 - 2.0*np.cos(2.0*math.pi*(myg.x2d-t))*np.sin(2.0*math.pi*(myg.y2d-t))
v_exact = 1.0 + 2.0*np.sin(2.0*math.pi*(myg.x2d-t))*np.cos(2.0*math.pi*(myg.y2d-t))

# error
udiff = u_exact - u
vdiff = v_exact - v

print "errors: ", myg.norm(udiff), myg.norm(vdiff)

