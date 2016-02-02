#!/bin/sh

# this script builds each of the shared-object libraries that
# interface Fortran with python for some lower-level pyro routines.
# f2py is used.

# set the python interpreter to use.  If no PYTHON variable is
# set, then default to python.  You can use python3, for example,
# by doing:
# PYTHON=python3 ./mk.sh
: ${PYTHON:=python2}

if [ "$1" == "clean" ]; then

    rm -rf mesh/*.so
    rm -rf multigrid/*.so
    rm -rf incompressible/*.so
    rm -rf compressible/*.so
    rm -rf compressible_gr/*.so
    rm -rf lm_atm/*.so
    rm -rf lm_combustion/*.so
    rm -rf lm_gr/*.so
    find . -name "*.pyc" -exec rm -f {} \;

else
    if [ "$1" == "debug" ]; then
	FFLAGS="-fbounds-check -fbacktrace -Wuninitialized -Wunused -ffpe-trap=invalid -finit-real=snan"
    else
	FFLAGS="-C"
    fi

    for d in mesh multigrid incompressible compressible compressible_gr lm_atm lm_gr lm_combustion
    do
	cd ${d}
	${PYTHON} setup.py config_fc --f90flags "${FFLAGS}" build_ext --inplace
	cd ..
    done
fi
