# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

cext = Extension("cons_to_prim", ["cons_to_prim.pyx"])

setup(ext_modules=cythonize(cext), include_dirs=[numpy.get_include()])

from numpy.distutils.core import setup, Extension

extra_link_args=["-L/usr/lib/x86_64-linux-gnu/4.8"]

ext = [Extension("LM_gr_interface_f",
                ["LM_gr_interface_f.f90"], f2py_options=["-L/home/alice/anaconda3/pkgs/libgcc-5.2.0-0/lib"], extra_link_args=extra_link_args)]#,
                #Extension("LM_gr_interface_sph_f",
                #["LM_gr_interface_sph_f.f90"])]

setup(ext_modules=ext)
