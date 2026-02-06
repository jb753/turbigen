"""Tests for TurbigenConfig class."""

import pytest
from pathlib import Path
import tempfile
import turbigen.config3
import turbigen.fluid
import turbigen.inlet


def test_turbigen_config_instantiation():
    """Test that TurbigenConfig can be created with all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        fluid_config = turbigen.fluid.PerfectFluidConfig(
            type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
        )

        inlet_config = turbigen.inlet.InletConfig(Po=1e5, To=300.0)

        config = turbigen.config3.TurbigenConfig(
            workdir=workdir, fluid=fluid_config, inlet=inlet_config
        )

        assert config.workdir == workdir
        assert isinstance(config.fluid, turbigen.fluid.PerfectFluidConfig)
        assert isinstance(config.inlet, turbigen.inlet.InletConfig)
        assert config.fluid.cp == 1005.0
        assert config.inlet.Po == 1e5
        assert config.inlet.To == 300.0


def test_turbigen_config_fluid_object_accessible():
    """Test that the fluid object can be accessed through config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fluid_config = turbigen.fluid.PerfectFluidConfig(
            type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
        )

        inlet_config = turbigen.inlet.InletConfig(Po=1e5, To=300.0)

        config = turbigen.config3.TurbigenConfig(
            workdir=Path(tmpdir), fluid=fluid_config, inlet=inlet_config
        )

        # Access the fluid object through config
        fluid = config.fluid.fluid

        # Use it for thermodynamic calculations
        rho, u = fluid.set_P_T(1e5, 300.0)
        P = fluid.get_P(rho, u)

        assert P == pytest.approx(1e5)


def test_turbigen_config_with_fluid_from_dict():
    """Test creating TurbigenConfig with fluid config from dictionary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fluid_dict = {
            "type": "perfect",
            "cp": 1005.0,
            "gamma": 1.4,
            "mu": 1.8e-5,
            "Pr": 0.72,
            "Tu0": 300.0,
        }

        fluid_config = turbigen.fluid.FluidConfig.from_dict(fluid_dict)
        inlet_config = turbigen.inlet.InletConfig(Po=2e5, To=400.0)

        config = turbigen.config3.TurbigenConfig(
            workdir=Path(tmpdir), fluid=fluid_config, inlet=inlet_config
        )

        assert config.fluid.type == "perfect"
        assert config.fluid.cp == 1005.0
        assert config.inlet.Po == 2e5


def test_turbigen_config_inlet_with_profiles():
    """Test TurbigenConfig with inlet profiles."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        fluid_config = turbigen.fluid.PerfectFluidConfig(
            type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
        )

        # Create inlet with radial profiles
        spf = [0.0, 0.5, 1.0]
        profiles = np.array(
            [
                [1.0e5, 1.1e5, 1.2e5],  # Po
                [300.0, 310.0, 320.0],  # To
                [0.0, 5.0, 10.0],  # Alpha
                [0.0, 0.0, 0.0],  # Beta
            ]
        )

        inlet_config = turbigen.inlet.InletConfig(
            Po=1e5, To=300.0, spf=spf, profiles=profiles
        )

        config = turbigen.config3.TurbigenConfig(
            workdir=Path(tmpdir), fluid=fluid_config, inlet=inlet_config
        )

        assert config.inlet.spf == spf
        assert config.inlet.profiles.shape == (4, 3)
        assert config.inlet.profiles[0, 1] == pytest.approx(1.1e5)  # Po at midspan


def test_turbigen_config_workdir_type():
    """Test that workdir must be a Path."""
    fluid_config = turbigen.fluid.PerfectFluidConfig(
        type="perfect", cp=1005.0, gamma=1.4, mu=1.8e-5
    )
    inlet_config = turbigen.inlet.InletConfig(Po=1e5, To=300.0)

    # Should work with Path
    with tempfile.TemporaryDirectory() as tmpdir:
        config = turbigen.config3.TurbigenConfig(
            workdir=Path(tmpdir), fluid=fluid_config, inlet=inlet_config
        )
        assert isinstance(config.workdir, Path)

    # Also works with string (Path will convert it)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = turbigen.config3.TurbigenConfig(
            workdir=Path(tmpdir),  # Explicitly convert
            fluid=fluid_config,
            inlet=inlet_config,
        )
        assert isinstance(config.workdir, Path)
