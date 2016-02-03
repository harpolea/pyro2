# this works for python 2 or 3 directly.  To build, do:
#
# python2 setup.py build_ext --inplace

from numpy.distutils.core import setup, Extension

extra_link_args=[]

ext = Extension("interface_f",
                ["interface_f.f90"])

setup(ext_modules=[ext])

# cythoning does not speed up function (?)
from distutils.core import setup, Extension
from Cython.Build import cythonize

cext = Extension("cons_to_prim", ["cons_to_prim.pyx"])

setup(ext_modules=cythonize(cext))