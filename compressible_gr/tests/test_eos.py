"""
Test the equation of state.
"""

import numpy as np
import compressible_gr.eos as eos
from numpy.testing import assert_allclose

def test_pres():
    gamma = 5./3.

    assert_allclose(0., eos.pres(gamma, 0., 5.))
    assert_allclose(0., eos.pres(gamma, 5., 0.))

    assert_allclose(2./3., eos.pres(gamma, 1., 1.))

def test_dens():
    gamma = 5./3.

    assert_allclose(0., eos.dens(gamma, 0., 1.))
    assert_allclose(1.5, eos.dens(gamma, 1., 1.))

def test_rhoe():
    gamma = 5./3.

    assert_allclose(0., eos.rhoe(gamma, 0.))
    assert_allclose(1.5, eos.rhoe(gamma, 1.))
