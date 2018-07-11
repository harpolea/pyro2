#!/bin/bash

# this script builds the shared-object libraries that interface
# Fortran with python for some lower-level pyro routines.  f2py is
# used.
#
# use `./mk.sh clean` to clear all the build files
#
# set the python interpreter to use.  If no PYTHON variable is
# set, then default to python3.  You can use python2, for example,
# by doing:
# PYTHON=python2 ./mk.sh
: ${PYTHON:=python3}

if [ "$1" == "clean" ]; then

    rm -rf mesh/*.so
    rm -rf incompressible/*.so
    rm -rf compressible/*.so
    rm -rf lm_atm/*.so
    rm -rf swe/*.so
    rm -rf advection_fv4/*.so
    find . -name "*.pyc" -exec rm -f {} \;
    find . -type d -name "__pycache__" -exec rm -rf {} \;
    find . -type d -name "build" -exec rm -rf {} \;
    find . -name "*wrapper.cpp" -exec rm -f {} \;

elif [ "$1" == "cython" ]; then
    # compiler looks for changes in the .cpp files rather than
    # the .pyx files, so delete them to force recompilation.
    # This would work better with a makefile where the file
    # dependencies could be made explicit
    find . -name "*wrapper.cpp" -exec rm -f {} \;

    ${PYTHON} setup_cython.py build_ext --inplace

else
    if [ "$1" == "debug" ]; then
	FFLAGS="-fbounds-check -fbacktrace -Wuninitialized -Wunused -ffpe-trap=invalid -finit-real=snan"
    else
	FFLAGS="-C"
    fi

    ${PYTHON} setup.py config_fc --f90flags "${FFLAGS}" build_ext

fi
