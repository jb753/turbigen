"""Tests for thickness distribution classes."""

import numpy as np
import pytest
from turbigen.thickness import BaseThickness, Taylor
from turbigen import util


# All thickness subclasses for parameterized testing
THICKNESS_CLASSES = [
    Taylor,
]

# Test parameter sets for each subclass
# Taylor parameters: [R_LE, t_max, m_tmax, kappa_max, t_te, tanwedge]
THICKNESS_PARAMS = {
    "Taylor": [
        [0.05, 0.1, 0.3, 0.0, 0.02, 0.1],
        [0.02, 0.05, 0.3, 0.0, 0.0, 0.1],
    ],
}


@pytest.fixture(params=THICKNESS_CLASSES, ids=lambda cls: cls.__name__)
def thickness_class(request):
    """Fixture providing all thickness classes."""
    return request.param


@pytest.fixture
def thickness_params(thickness_class):
    """Fixture providing test parameters for each thickness class."""
    return THICKNESS_PARAMS[thickness_class.__name__]


class TestBaseFunctionality:
    """Test basic functionality common to all thickness subclasses."""

    def test_all_subclasses_covered(self):
        """Test that all BaseThickness subclasses are included in THICKNESS_CLASSES."""
        all_subclasses = set(BaseThickness.__subclasses__())
        tested_classes = set(THICKNESS_CLASSES)

        missing_classes = all_subclasses - tested_classes
        assert not missing_classes, (
            f"Missing thickness classes in tests: {missing_classes}"
        )

        # Also verify no extra classes in test list
        extra_classes = tested_classes - all_subclasses
        assert not extra_classes, f"Extra classes in test list: {extra_classes}"

    def test_initialization_valid_params(self, thickness_class, thickness_params):
        """Test initialization with valid parameter vectors."""
        for params in thickness_params:
            thick = thickness_class(params)
            assert hasattr(thick, "q_thick")
            assert len(thick.q_thick) == len(params)
            assert np.allclose(thick.q_thick, params)

    def test_initialization_array_reshape(self, thickness_class, thickness_params):
        """Test that parameter vectors are properly reshaped to 1D."""
        params = thickness_params[0]
        # Test various input shapes
        shapes_to_test = [
            np.array(params),  # 1D array
            np.array(params).reshape(-1, 1),  # Column vector
            np.array(params).reshape(1, -1),  # Row vector
        ]

        for param_array in shapes_to_test:
            thick = thickness_class(param_array)
            assert thick.q_thick.ndim == 1
            assert len(thick.q_thick) == len(params)

    def test_property_access(self, thickness_class, thickness_params):
        """Test access to key properties."""
        for params in thickness_params:
            thick = thickness_class(params)

            # All subclasses should have these properties
            assert hasattr(thick, "t_max")
            assert hasattr(thick, "R_LE")
            assert hasattr(thick, "t_te")

            # Properties should return numeric values
            assert isinstance(thick.t_max, (int, float, np.number))
            assert isinstance(thick.R_LE, (int, float, np.number))
            assert isinstance(thick.t_te, (int, float, np.number))

    def test_scale_method(self, thickness_class, thickness_params):
        """Test that scale method exists and modifies appropriate parameters."""
        for params in thickness_params:
            thick = thickness_class(params)

            # Get original values
            orig_t_max = thick.t_max
            orig_R_LE = thick.R_LE
            orig_t_te = thick.t_te

            # Scale by factor
            scale_factor = 2.0
            thick.scale(scale_factor)

            # Check that scalable properties changed appropriately
            assert np.isclose(thick.t_max, orig_t_max * scale_factor)
            assert np.isclose(thick.R_LE, orig_R_LE * scale_factor)
            assert np.isclose(thick.t_te, orig_t_te * scale_factor)


@pytest.mark.parametrize(
    "thickness_class,param_idx",
    [
        (cls, idx)
        for cls in THICKNESS_CLASSES
        for idx in range(len(THICKNESS_PARAMS[cls.__name__]))
    ],
    ids=[
        f"{cls.__name__[0]}{idx}"
        for cls in THICKNESS_CLASSES
        for idx in range(len(THICKNESS_PARAMS[cls.__name__]))
    ],
)
class TestThickMethod:
    """Test thick method for all thickness subclasses."""

    @pytest.fixture
    def thickness_instance(self, thickness_class, param_idx):
        """Fixture providing all thickness instances with all parameter combinations."""
        params = THICKNESS_PARAMS[thickness_class.__name__][param_idx]
        return thickness_class(params)

    def test_thick_domain_validation_scalar(self, thickness_instance):
        """Test thick with scalar inputs in valid domain [0,1]."""
        test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for m in test_points:
            result = thickness_instance.thick(m)
            assert np.isscalar(result), f"thick({m}) should return scalar"
            assert np.isfinite(result), f"thick({m}) should be finite"
            assert result >= 0, f"thick({m}) should be non-negative"

    def test_thick_domain_validation_array(self, thickness_instance):
        """Test thick with array inputs in valid domain [0,1]."""
        m_arrays = [
            np.array([0.0, 0.5, 1.0]),
            np.linspace(0, 1, 11),
            np.array([[0.0, 0.5], [0.75, 1.0]]),  # 2D array
        ]

        for m in m_arrays:
            result = thickness_instance.thick(m)
            assert result.shape == m.shape, f"Output shape should match input shape"
            assert np.all(np.isfinite(result)), "All thick values should be finite"
            assert np.all(result >= 0), "All thick values should be non-negative"

    def test_thick_domain_invalid_scalar(self, thickness_instance):
        """Test thick raises error for scalar values outside [0,1]."""
        invalid_values = [-0.1, -1.0, 1.1, 2.0]

        for m in invalid_values:
            with pytest.raises((ValueError, RuntimeError, IndexError)):
                thickness_instance.thick(m)

    def test_thick_domain_invalid_array(self, thickness_instance):
        """Test thick raises error for array values outside [0,1]."""
        invalid_arrays = [
            np.array([-0.1, 0.5, 1.0]),  # Contains negative
            np.array([0.0, 0.5, 1.1]),  # Contains > 1
            np.array([-0.5, 1.5]),  # Both bounds exceeded
            np.linspace(-0.2, 1.2, 5),  # Range exceeds bounds
        ]

        for m in invalid_arrays:
            with pytest.raises((ValueError, RuntimeError, IndexError)):
                thickness_instance.thick(m)

    def test_thick_boundary_conditions(self, thickness_instance):
        """Test thick boundary conditions: thick(0)=0, thick(1)=t_te/2."""
        # Test leading edge condition: thick(0) = 0
        thick_le = thickness_instance.thick(0.0)
        assert np.isclose(thick_le, 0.0, atol=1e-10), "thick(0) should be 0"

        # Test trailing edge condition: thick(1) = t_te/2
        thick_te = thickness_instance.thick(1.0)
        expected_te = thickness_instance.t_te / 2.0
        assert np.isclose(thick_te, expected_te, atol=1e-10), (
            f"thick(1) should be t_te/2 = {expected_te}, got {thick_te}"
        )

    def test_thick_maximum_value(self, thickness_instance):
        """Test that max(thick) equals t_max property."""
        # Sample thickness at many points to find maximum
        m = np.linspace(0, 1, 5000)
        thick_values = thickness_instance.thick(m)
        max_thick = np.max(thick_values)

        # Should match t_max property within tolerance
        assert np.isclose(max_thick, thickness_instance.t_max, atol=1e-6), (
            f"max(thick) = {max_thick} should equal t_max = {thickness_instance.t_max}"
        )

    def test_thick_continuity(self, thickness_instance):
        """Test continuity of thickness distribution."""
        # Use fine grid to check for sudden jumps
        m_fine = util.cluster_cosine(5001)
        thick_values = thickness_instance.thick(m_fine)

        # Check for no sudden jumps (basic continuity check)
        jump = np.abs(np.diff(thick_values))
        max_jump = np.max(jump)

        # Threshold should be reasonable for thickness distributions
        max_allowed_jump = thickness_instance.t_max * 0.01
        assert max_jump < max_allowed_jump, (
            f"Thickness should be continuous (max jump: {max_jump})"
        )

    def test_thick_monotonicity_before_max(self, thickness_instance):
        """Test that thickness generally increases before maximum thickness point."""
        # This is a heuristic test - not all thickness distributions are strictly monotonic
        # But they should generally increase toward the maximum thickness point

        # Find approximate location of maximum thickness
        m_search = np.linspace(0, 1, 101)
        thick_search = thickness_instance.thick(m_search)
        max_idx = np.argmax(thick_search)
        m_max_approx = m_search[max_idx]

        # Sample points before the maximum
        if m_max_approx > 0.1:  # Only test if max is not too close to LE
            m_before = np.linspace(0.05, m_max_approx - 0.05, 10)
            thick_before = thickness_instance.thick(m_before)

            # Check that thickness generally increases (allowing some small decreases)
            differences = np.diff(thick_before)
            decreasing_count = np.sum(differences < -1e-8)

            # Allow up to 20% of points to have small decreases (flexibility for splines)
            assert decreasing_count <= len(differences) * 0.2, (
                "Thickness should generally increase before maximum"
            )

    def test_output_type_matches_input_type(self, thickness_instance):
        """Test that output type and shape matches input type and shape."""
        # Test scalar input returns scalar
        m_scalar = 0.5
        thick_scalar = thickness_instance.thick(m_scalar)

        assert np.isscalar(thick_scalar), "thick should return scalar for scalar input"
        assert isinstance(thick_scalar, (int, float, np.number)), (
            "thick should return numeric scalar"
        )

        # Test various array shapes
        test_shapes = [
            (3,),  # 1D array
            (2, 3),  # 2D array
            (2, 2, 2),  # 3D array
            (1,),  # Single element array
            (1, 1),  # Single element 2D array
        ]

        for shape in test_shapes:
            m_array = np.linspace(0, 1, np.prod(shape)).reshape(shape)
            thick_array = thickness_instance.thick(m_array)

            assert thick_array.shape == m_array.shape, (
                f"thick shape mismatch for {shape}"
            )
            assert isinstance(thick_array, np.ndarray), (
                f"thick should return ndarray for shape {shape}"
            )

    def test_thick_physical_constraints(self, thickness_instance):
        """Test that thickness satisfies physical constraints."""
        m = np.linspace(0, 1, 51)
        thick_values = thickness_instance.thick(m)

        # All thickness values should be non-negative
        assert np.all(thick_values >= 0), "All thickness values should be non-negative"

        # Thickness should not exceed maximum thickness (with small tolerance)
        assert np.all(thick_values <= thickness_instance.t_max + 1e-10), (
            "Thickness should not exceed t_max"
        )

        # Leading edge should be exactly zero
        assert np.isclose(thick_values[0], 0.0, atol=1e-12), (
            "Leading edge thickness should be exactly zero"
        )

    def test_broadcasting_mixed_shapes(self, thickness_instance):
        """Test broadcasting behavior with different input shapes."""
        # Test with list input (should be converted to array)
        m_list = [0.0, 0.5, 1.0]
        thick_list = thickness_instance.thick(m_list)

        assert len(thick_list) == 3, "List input should work"
        assert np.all(thick_list >= 0), "List input should give valid results"
