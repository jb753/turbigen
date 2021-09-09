"""Tests for the turbogen module"""
import numpy as np
from turbogen.make_design import *

# Set up test data

# Ranges of velocity triangle parameters covering the classic Smith chart
phi = np.linspace(0.4,1.2,7)
psi = np.linspace(0.8,2.4,7)

# "Reasonable" range of reaction (usually close to Lam = 0.5 in gas turbines)
# Know limitation: does not converge for very high reaction Lam > 0.8
Lam = np.linspace(0.3,0.7,5)

# Other parameters
Al1 = 0.
Ma2 = 0.6
ga = 1.33
eta = 0.9

# Begin test functions

def test_target_Lam():
    """Check target reaction is achieved by the yaw angle iteration."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                        phii, psii, Lami, Al1, Ma2, ga, eta
                        )
                assert np.isclose(stg.Lam, Lami)

def test_valid():
    """Check that output data is always physically sensible."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                        phii, psii, Lami, Al1, Ma2, ga, eta
                        )
                # No nans or infinities
                for xi in stg:
                    assert np.all(np.isfinite(xi))
                # All variables excluding flow angles should be non-negative
                for vi, xi in stg._asdict().items():
                    if not 'Al' in vi:
                        assert np.all(np.array(xi)>=0.)
                # Flow angles less than 90 degrees
                for vi in ['Al','Al_rel']:
                    assert np.all(np.abs(getattr(stg,vi))<90.)
                # No diverging annuli
                assert np.all(np.array(stg.Ax_Axin)>=1.)


