# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext
#
# Note the setup.cfg directs the build to be done in-place.

from numpy.distutils.core import setup, Extension
import numpy
from Cython.Build import build_ext

# ext_modules = [Extension("compressible.interface_f", ["compressible/interface_f.f90"]),
#                Extension("advection_fv4.interface_f", ["advection_fv4/interface_states.f90"]),
#                Extension("lm_atm.LM_atm_interface_f", ["lm_atm/LM_atm_interface_f.f90"]),
#                Extension("incompressible.incomp_interface_f", ["incompressible/incomp_interface_f.f90"]),
#                Extension("swe.interface_f", ["swe/interface_f.f90"])]

# ext_modules = cythonize("compressible/interface.pyx", annotate=True)

ext_modules = [Extension("compressible.interface_c",
               sources=["compressible/interface_wrapper.pyx",
               "compressible/c_interface.c"],
               include_dirs=[numpy.get_include()]),
               Extension("incompressible.incomp_interface_c",
              sources=["incompressible/incomp_interface_wrapper.pyx",
              "incompressible/c_incomp_interface.c"],
              include_dirs=[numpy.get_include()]),
              Extension("advection_fv4.interface_c",
                         sources=["advection_fv4/interface_states_wrapper.pyx",
                         "advection_fv4/c_interface_states.c"],
                         include_dirs=[numpy.get_include()])]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules)
