# this works for python 2 or 3 directly.  To build, do:
#
# python2 setup.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

cext = Extension("cons_to_prim", ["cons_to_prim.pyx"])#,
#        include_dirs = ["."],
#        extra_compile_args = ["-O3", "-Wall"],
#        extra_link_args = ['-g'])]

setup(ext_modules=cythonize(cext), include_dirs=[numpy.get_include()])

from numpy.distutils.core import setup, Extension

extra_link_args=[]

ext = Extension("interface_f",
                ["interface_f.f90"])

setup(ext_modules=[ext])
