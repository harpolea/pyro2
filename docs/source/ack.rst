Acknowledgments
===============

Pyro developed by (in alphabetical order):

  * Alice Harpole
  * Ian Hawke
  * Michael Zingale


You are free to use this code and the accompanying notes in your
classes. Please credit "pyro development team" for the code, and
*please send a note to the pyro-help e-mail list describing how you
use it, so we can keep track of it (and help justify the development
effort).*

If you use pyro in a publication, please cite it using this bibtex
citation::

  @ARTICLE{pyro:2014,
     author = {{Zingale}, M.},
      title = "{pyro: A teaching code for computational astrophysical hydrodynamics}",
    journal = {Astronomy and Computing},
  archivePrefix = "arXiv",
     eprint = {1306.6883},
   primaryClass = "astro-ph.IM",
   keywords = {Hydrodynamics, Methods: numerical},
       year = 2014,
      month = oct,
     volume = 6,
      pages = {52--62},
        doi = {10.1016/j.ascom.2014.07.003},
     adsurl = {http://adsabs.harvard.edu/abs/2014A%26C.....6...52Z},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }

pyro benefited from numerous useful discussions with Ann Almgren, John
Bell, and Andy Nonaka.


History
=======

The original pyro code was written in 2003-4 to help developmer
Zingale understand these methods for himself. It was originally written
using the Numeric array package and handwritten C extensions for the
compute-intensive kernels.  It was ported to numarray when that
replaced Numeric, and continued to use C extensions.  This version
"pyro2" was resurrected beginning in 2012 and rewritten for numpy
using f2py, and brought up to date.

