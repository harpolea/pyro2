# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext --inplace

from numpy.distutils.core import setup, Extension

extra_link_args = ["-L/Developer/SDKs/MacOSX10.7.sdk/usr/lib"]

ext = Extension("LM_atm_interface_f",
                ["LM_atm_interface_f.f90"])

setup(ext_modules=[ext])
