"""Tests for fluid configuration classes."""

import pytest
import numpy as np
import dataclasses
import turbigen.fluid
import ember.fluid


def test_perfect_fluid_config_instantiation():
    """Test that PerfectFluidConfig creates with required and default values."""
    config = turbigen.fluid.PerfectFluidConfig(
        type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
    )

    assert config.type == "perfect"
    assert config.cp == 1005.0
    assert config.gamma == 1.4
    assert config.mu == 1.8e-5
    assert config.Pr == 0.7

    # Check that fluid object was created
    assert isinstance(config.fluid, ember.fluid.PerfectFluid)


def test_perfect_fluid_config_creates_working_fluid():
    """Test that the created fluid object works correctly."""
    config = turbigen.fluid.PerfectFluidConfig(
        type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
    )

    # Use the fluid to do thermodynamic calculations
    P, T = 1e5, 300.0
    rho, u = config.fluid.set_P_T(P, T)

    # Check basic perfect gas relation
    rgas = config.fluid.get_Rgas(rho, u)
    P_check = config.fluid.get_P(rho, u)
    T_check = config.fluid.get_T(rho, u)

    assert np.isclose(P_check, P)
    assert np.isclose(T_check, T)
    assert np.isclose(P, rho * rgas * T)


def test_invalid_fluid_type_raises_error():
    """Test that invalid fluid type raises an error."""
    with pytest.raises(AssertionError):
        # This should fail in __post_init__ when validating the type
        # PerfectFluidConfig requires type="perfect"
        turbigen.fluid.PerfectFluidConfig(
            type="wrong_type", cp=1005.0, gamma=1.4, mu=1.8e-5
        )


def test_fluid_config_immutability():
    """Test that fluid config is frozen and cannot be modified after creation."""
    config = turbigen.fluid.PerfectFluidConfig(
        type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
    )

    # Attempting to modify should raise FrozenInstanceError
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.cp = 2000.0

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.gamma = 1.5

    # The fluid object should remain unchanged
    rho, u = config.fluid.set_P_T(1e5, 300.0)
    cp = config.fluid.get_cp(rho, u)
    assert np.isclose(cp, 1005.0)


def test_different_configs_have_independent_fluids():
    """Test that different configs create independent fluid instances."""
    config1 = turbigen.fluid.PerfectFluidConfig(
        type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
    )
    config2 = turbigen.fluid.PerfectFluidConfig(
        type="perfect", cp=1200.0, gamma=1.3, mu=2.0e-5
    )

    assert config1.fluid is not config2.fluid

    # Check that fluids have different properties
    rho, u = 1.0, 1000.0
    cp1 = config1.fluid.get_cp(rho, u)
    cp2 = config2.fluid.get_cp(rho, u)

    assert np.isclose(cp1, 1005.0)
    assert np.isclose(cp2, 1200.0)


def test_from_dict_creates_correct_config():
    """Test that from_dict factory method creates the right config type."""
    config_dict = {
        "type": "perfect",
        "cp": 1005.0,
        "gamma": 1.4,
        "mu": 1.8e-5,
        "Pr": 0.72,
    }

    config = turbigen.fluid.FluidConfig.from_dict(config_dict)

    # Check it created the right type
    assert isinstance(config, turbigen.fluid.PerfectFluidConfig)
    assert config.type == "perfect"
    assert config.cp == 1005.0
    assert config.gamma == 1.4

    # Check the fluid works
    P, T = 1e5, 300.0
    rho, u = config.fluid.set_P_T(P, T)
    P_check = config.fluid.get_P(rho, u)
    assert np.isclose(P_check, P)


def test_from_dict_with_invalid_type_raises_error():
    """Test that from_dict raises error for unknown fluid type."""
    config_dict = {"type": "unknown_fluid", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5}

    with pytest.raises(ValueError, match="Unknown fluid type"):
        turbigen.fluid.FluidConfig.from_dict(config_dict)


def test_from_dict_does_not_modify_input():
    """Test that from_dict doesn't modify the input dictionary."""
    config_dict = {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5}

    original_dict = config_dict.copy()
    turbigen.fluid.FluidConfig.from_dict(config_dict)

    # Input should be unchanged
    assert config_dict == original_dict
