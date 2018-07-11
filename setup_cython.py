# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext
#
# Note the setup.cfg directs the build to be done in-place.

from numpy.distutils.core import setup, Extension
import numpy
from Cython.Build import build_ext

ext_modules = [Extension("compressible.interface_c",
                         sources=["compressible/interface_wrapper.pyx",
                                  "compressible/c_interface.cpp"],
                         include_dirs=[numpy.get_include()], language="c++"),
               Extension("incompressible.incomp_interface_c",
                         sources=["incompressible/incomp_interface_wrapper.pyx",
                                  "incompressible/c_incomp_interface.cpp"],
                         include_dirs=[numpy.get_include()], language="c++"),
               Extension("advection_fv4.interface_c",
                         sources=["advection_fv4/interface_states_wrapper.pyx",
                                  "advection_fv4/c_interface_states.cpp"],
                         include_dirs=[numpy.get_include()], language="c++"),
               Extension("lm_atm.LM_atm_interface_c",
                         sources=["lm_atm/LM_atm_interface_wrapper.pyx",
                                  "lm_atm/c_LM_atm_interface.cpp"],
                         include_dirs=[numpy.get_include()], language="c++"),
               Extension("swe.interface_c",
                         sources=["swe/interface_wrapper.pyx",
                                  "swe/c_interface.cpp"],
                         include_dirs=[numpy.get_include()], language="c++")]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
