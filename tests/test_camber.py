"""Tests for camber line classes."""

import numpy as np
import pytest
from turbigen import camber
import itertools as it

# Global tolerances
ATOL_CHI = 1e-2
ATOL_DYDM = 1e-3

# All camber subclasses for parameterized testing
CAMBER_CLASSES = [
    camber.Quartic,
    camber.Taylor,
    camber.Quadratic,
    camber.TaylorQuadratic,
]

tan30 = np.tan(np.radians(30))
tan60 = np.tan(np.radians(60))
tan89 = np.tan(np.radians(89))
CHI_LE_TE = [0.0, tan30, -tan30, tan89, -tan89]

# Test parameter sets for each subclass
# Format: (tanchi_LE, tanchi_TE, *additional_params)

CAMBER_PARAMS_ADD = {
    "Quartic": [
        [1.0, 1.0, 0.0],
        [2.0, 0.5, -0.5],
        [0.5, 2.0, 0.5],
    ],
    "Taylor": [
        [1.0, 1.0, 0.0],
        [2.0, 0.5, -0.2],
        [0.5, 2.0, 0.2],
    ],
    "Quadratic": [
        [0.0],
        [-1.0],
        [1.0],
    ],
    "TaylorQuadratic": [
        [0.0],
        [-1.0],
        [1.0],
    ],
}

# Generate CAMBER_PARAMS: all permutations of 2 choices from CHI_LE_TE
# combined with each choice from CAMBER_PARAMS_ADD
CAMBER_PARAMS = {}

for class_name, additional_params in CAMBER_PARAMS_ADD.items():
    params_list = []

    # All permutations of 2 choices from CHI_LE_TE (with replacement)
    for tanchi_le, tanchi_te in it.product(CHI_LE_TE, repeat=2):
        # Combine with each choice of additional parameters
        for add_params in additional_params:
            full_params = [tanchi_le, tanchi_te] + add_params
            params_list.append(full_params)

    CAMBER_PARAMS[class_name] = params_list


@pytest.fixture(params=CAMBER_CLASSES, ids=lambda cls: cls.__name__)
def camber_class(request):
    """Fixture providing all camber classes."""
    return request.param


@pytest.fixture
def camber_params(camber_class):
    """Fixture providing test parameters for each camber class."""
    return CAMBER_PARAMS[camber_class.__name__]


class TestBaseFunctionality:
    """Test basic functionality common to all camber subclasses."""

    def test_all_subclasses_covered(self):
        """Test that all BaseCamber subclasses are included in CAMBER_CLASSES."""
        all_subclasses = set(camber.BaseCamber.__subclasses__())
        tested_classes = set(CAMBER_CLASSES)
        
        missing_classes = all_subclasses - tested_classes
        assert not missing_classes, f"Missing camber classes in tests: {missing_classes}"
        
        # Also verify no extra classes in test list
        extra_classes = tested_classes - all_subclasses
        assert not extra_classes, f"Extra classes in test list: {extra_classes}"

    def test_initialization_valid_params(self, camber_class, camber_params):
        """Test initialization with valid parameter vectors."""
        for params in camber_params:
            cam = camber_class(params)
            assert hasattr(cam, "q_camber")
            assert len(cam.q_camber) == len(params)
            assert np.allclose(cam.q_camber, params)

    def test_initialization_array_reshape(self, camber_class, camber_params):
        """Test that parameter vectors are properly reshaped to 1D."""
        params = camber_params[0]
        # Test various input shapes
        shapes_to_test = [
            np.array(params),  # 1D array
            np.array(params).reshape(-1, 1),  # Column vector
            np.array(params).reshape(1, -1),  # Row vector
        ]

        for param_array in shapes_to_test:
            cam = camber_class(param_array)
            assert cam.q_camber.ndim == 1
            assert len(cam.q_camber) == len(params)

    def test_property_access_tanchi(self, camber_class, camber_params):
        """Test access to tangent of camber angles."""
        for params in camber_params:
            cam = camber_class(params)
            assert cam.tanchi_LE == params[0]
            assert cam.tanchi_TE == params[1]

    def test_property_access_chi_degrees(self, camber_class, camber_params):
        """Test access to camber angles in degrees."""
        for params in camber_params:
            cam = camber_class(params)
            expected_chi_LE = np.degrees(np.arctan(params[0]))
            expected_chi_TE = np.degrees(np.arctan(params[1]))

            assert np.isclose(cam.chi_LE, expected_chi_LE)
            assert np.isclose(cam.chi_TE, expected_chi_TE)

    def test_property_access_deltas(self, camber_class, camber_params):
        """Test access to camber angle differences."""
        for params in camber_params:
            cam = camber_class(params)
            expected_Dchi = cam.chi_TE - cam.chi_LE
            expected_Dtanchi = params[1] - params[0]

            assert np.isclose(cam.Dchi, expected_Dchi)
            assert np.isclose(cam.Dtanchi, expected_Dtanchi)

    def test_hash_consistency(self, camber_class, camber_params):
        """Test that identical parameters produce identical hashes."""
        for params in camber_params:
            cam1 = camber_class(params)
            cam2 = camber_class(params)
            assert hash(cam1) == hash(cam2)

    def test_hash_different_params(self, camber_class, camber_params):
        """Test that different parameters produce different hashes."""
        if len(camber_params) > 1:
            cam1 = camber_class(camber_params[0])
            cam2 = camber_class(camber_params[1])
            assert hash(cam1) != hash(cam2)

    def test_load_camber_function(self, camber_class):
        """Test that classes can be loaded by string name."""
        class_name = camber_class.__name__
        loaded_class = camber.load_camber(class_name)
        assert loaded_class == camber_class

    def test_load_camber_invalid_name(self):
        """Test that invalid class names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown camber type"):
            camber.load_camber("NonexistentCamber")


@pytest.mark.parametrize(
    "camber_class,param_idx",
    [(cls, idx) for cls in CAMBER_CLASSES for idx in range(len(CAMBER_PARAMS[cls.__name__]))],
    ids=[f"{cls.__name__[0]}{idx}" for cls in CAMBER_CLASSES for idx in range(len(CAMBER_PARAMS[cls.__name__]))]
)
class TestCoreMethods:
    """Test core methods chi_hat, chi, and dydm common to all camber subclasses."""

    @pytest.fixture
    def camber_instance(self, camber_class, param_idx):
        """Fixture providing all camber instances with all parameter combinations."""
        params = CAMBER_PARAMS[camber_class.__name__][param_idx]
        return camber_class(params)

    def test_chi_hat_domain_validation_scalar(self, camber_instance):
        """Test chi_hat with scalar inputs in valid domain [0,1]."""
        test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for m in test_points:
            result = camber_instance.chi_hat(m)
            assert np.isscalar(result), f"chi_hat({m}) should return scalar"
            assert np.isfinite(result), f"chi_hat({m}) should be finite"

    def test_chi_hat_domain_validation_array(self, camber_instance):
        """Test chi_hat with array inputs in valid domain [0,1]."""
        m_arrays = [
            np.array([0.0, 0.5, 1.0]),
            np.linspace(0, 1, 11),
            np.array([[0.0, 0.5], [0.75, 1.0]]),  # 2D array
        ]

        for m in m_arrays:
            result = camber_instance.chi_hat(m)
            assert result.shape == m.shape, f"Output shape should match input shape"
            assert np.all(np.isfinite(result)), "All chi_hat values should be finite"

    def test_chi_hat_domain_invalid_scalar(self, camber_instance):
        """Test chi_hat raises error for scalar values outside [0,1]."""
        invalid_values = [-0.1, -1.0, 1.1, 2.0]

        for m in invalid_values:
            with pytest.raises((ValueError, RuntimeError)):
                camber_instance.chi_hat(m)

    def test_chi_hat_domain_invalid_array(self, camber_instance):
        """Test chi_hat raises error for array values outside [0,1]."""
        invalid_arrays = [
            np.array([-0.1, 0.5, 1.0]),  # Contains negative
            np.array([0.0, 0.5, 1.1]),  # Contains > 1
            np.array([-0.5, 1.5]),  # Both bounds exceeded
            np.linspace(-0.2, 1.2, 5),  # Range exceeds bounds
        ]

        for m in invalid_arrays:
            with pytest.raises((ValueError, RuntimeError)):
                camber_instance.chi_hat(m)

    def test_chi_hat_boundary_conditions(self, camber_instance):
        """Test chi_hat boundary conditions: chi_hat(0)=0, chi_hat(1)=1."""
        # Test scalar boundary conditions
        assert np.isclose(camber_instance.chi_hat(0.0), 0.0, atol=1e-12), (
            "chi_hat(0) should be 0"
        )
        assert np.isclose(camber_instance.chi_hat(1.0), 1.0, atol=1e-12), (
            "chi_hat(1) should be 1"
        )

        # Test array boundary conditions
        m = np.array([0.0, 0.5, 1.0])
        chi_hat = camber_instance.chi_hat(m)
        assert np.isclose(chi_hat[0], 0.0, atol=1e-12), (
            "chi_hat(0) should be 0 in array"
        )
        assert np.isclose(chi_hat[-1], 1.0, atol=1e-12), (
            "chi_hat(1) should be 1 in array"
        )

    def test_chi_hat_monotonicity(self, camber_instance):
        """Test that chi_hat is monotonic for physical camber lines."""
        m = np.linspace(0, 1, 21)
        chi_hat = camber_instance.chi_hat(m)

        # chi_hat should be monotonically increasing (non-decreasing)
        dchi_hat = np.diff(chi_hat)
        assert np.all(dchi_hat >= -1e-12), "chi_hat should be monotonically increasing"

    def test_chi_scalar_input(self, camber_instance):
        """Test chi method with scalar inputs."""
        test_points = [0.0, 0.1, 0.5, 0.9, 1.0]

        for m in test_points:
            chi_result = camber_instance.chi(m)
            assert np.isscalar(chi_result), f"chi({m}) should return scalar"
            assert np.isfinite(chi_result), f"chi({m}) should be finite"

            # Result should be in reasonable range (degrees)
            assert -90 < chi_result < 90, f"chi({m}) should be in valid angle range"

    def test_chi_array_input(self, camber_instance):
        """Test chi method with array inputs."""
        m_arrays = [
            np.array([0.0, 0.5, 1.0]),
            np.linspace(0, 1, 11),
            np.array([[0.0, 0.5], [0.75, 1.0]]),
        ]

        for m in m_arrays:
            chi_result = camber_instance.chi(m)
            assert chi_result.shape == m.shape, "Output shape should match input shape"
            assert np.all(np.isfinite(chi_result)), "All chi values should be finite"
            assert np.all(np.abs(chi_result) < 90), (
                "All chi values should be in valid range"
            )

    def test_chi_domain_invalid_scalar(self, camber_instance):
        """Test chi raises error for scalar values outside [0,1]."""
        invalid_values = [-0.1, -1.0, 1.1, 2.0]

        for m in invalid_values:
            with pytest.raises((ValueError, RuntimeError)):
                camber_instance.chi(m)

    def test_chi_domain_invalid_array(self, camber_instance):
        """Test chi raises error for array values outside [0,1]."""
        invalid_arrays = [
            np.array([-0.1, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.1]),
            np.array([-0.5, 1.5]),
            np.linspace(-0.2, 1.2, 5),
        ]

        for m in invalid_arrays:
            with pytest.raises((ValueError, RuntimeError)):
                camber_instance.chi(m)

    def test_chi_boundary_values(self, camber_instance):
        """Test chi at boundary points matches expected angles."""
        chi_le = camber_instance.chi(0.0)
        chi_te = camber_instance.chi(1.0)

        # Should match the property values
        assert np.isclose(chi_le, camber_instance.chi_LE, atol=ATOL_CHI)
        assert np.isclose(chi_te, camber_instance.chi_TE, atol=ATOL_CHI)

    def test_dydm_scalar_input(self, camber_instance):
        """Test dydm method with scalar inputs."""
        test_points = [0.0, 0.1, 0.5, 0.9, 1.0]

        for m in test_points:
            dydm_result = camber_instance.dydm(m)
            assert np.isscalar(dydm_result), f"dydm({m}) should return scalar"
            assert np.isfinite(dydm_result), f"dydm({m}) should be finite"

    def test_dydm_array_input(self, camber_instance):
        """Test dydm method with array inputs."""
        m_arrays = [
            np.array([0.0, 0.5, 1.0]),
            np.linspace(0, 1, 11),
            np.array([[0.0, 0.5], [0.75, 1.0]]),
        ]

        for m in m_arrays:
            dydm_result = camber_instance.dydm(m)
            assert dydm_result.shape == m.shape, "Output shape should match input shape"
            assert np.all(np.isfinite(dydm_result)), "All dydm values should be finite"

    def test_dydm_domain_invalid_scalar(self, camber_instance):
        """Test dydm raises error for scalar values outside [0,1]."""
        invalid_values = [-0.1, -1.0, 1.1, 2.0]

        for m in invalid_values:
            with pytest.raises((ValueError, RuntimeError)):
                camber_instance.dydm(m)

    def test_dydm_domain_invalid_array(self, camber_instance):
        """Test dydm raises error for array values outside [0,1]."""
        invalid_arrays = [
            np.array([-0.1, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.1]),
            np.array([-0.5, 1.5]),
            np.linspace(-0.2, 1.2, 5),
        ]

        for m in invalid_arrays:
            with pytest.raises((ValueError, RuntimeError)):
                camber_instance.dydm(m)

    def test_dydm_boundary_values(self, camber_instance):
        """Test dydm at boundary points matches expected slopes."""
        dydm_le = camber_instance.dydm(0.0)
        dydm_te = camber_instance.dydm(1.0)

        # Should match the tangent values
        assert np.isclose(dydm_le, camber_instance.tanchi_LE, atol=ATOL_DYDM)
        assert np.isclose(dydm_te, camber_instance.tanchi_TE, atol=ATOL_DYDM)

    def test_chi_dydm_relationship(self, camber_instance):
        """Test relationship between chi and dydm: dydm = tan(chi_radians)."""
        m = np.linspace(0, 1, 11)
        chi_deg = camber_instance.chi(m)
        dydm = camber_instance.dydm(m)

        # Convert chi to radians and take tangent
        expected_dydm = np.tan(np.radians(chi_deg))

        assert np.allclose(dydm, expected_dydm, atol=ATOL_DYDM), (
            "dydm should equal tan(chi) within tolerance"
        )

    def test_continuity_fine_grid(self, camber_instance):
        """Test continuity of methods on fine grid."""
        # Skip continuity tests for extreme angle cases (> 60°)
        # These push polynomial models beyond reasonable operating range
        if abs(camber_instance.chi_LE) > 60 or abs(camber_instance.chi_TE) > 60:
            return

        m_fine = np.linspace(0, 1, 101)

        # All methods should produce smooth, continuous results
        chi_hat = camber_instance.chi_hat(m_fine)
        chi = camber_instance.chi(m_fine)
        dydm = camber_instance.dydm(m_fine)

        # Check for no sudden jumps (basic continuity check)
        max_chi_hat_diff = np.max(np.abs(np.diff(chi_hat)))
        max_chi_diff = np.max(np.abs(np.diff(chi)))
        max_dydm_diff = np.max(np.abs(np.diff(dydm)))

        # These thresholds are somewhat arbitrary but should catch major discontinuities
        assert max_chi_hat_diff < 0.15, "chi_hat should be continuous"
        assert max_chi_diff < 5.0, "chi should be continuous (max 5° jump)"
        assert max_dydm_diff < 1.0, "dydm should be continuous"

    def test_broadcasting_mixed_shapes(self, camber_instance):
        """Test broadcasting behavior with different input shapes."""
        # Test with list input (should be converted to array)
        m_list = [0.0, 0.5, 1.0]
        chi_hat_list = camber_instance.chi_hat(m_list)
        chi_list = camber_instance.chi(m_list)
        dydm_list = camber_instance.dydm(m_list)

        assert len(chi_hat_list) == 3, "List input should work"
        assert len(chi_list) == 3, "List input should work"
        assert len(dydm_list) == 3, "List input should work"

    def test_output_type_matches_input_type(self, camber_instance):
        """Test that output type and shape matches input type and shape."""
        # Test scalar input returns scalar
        m_scalar = 0.5
        chi_hat_scalar = camber_instance.chi_hat(m_scalar)
        chi_scalar = camber_instance.chi(m_scalar)
        dydm_scalar = camber_instance.dydm(m_scalar)

        assert np.isscalar(chi_hat_scalar), (
            "chi_hat should return scalar for scalar input"
        )
        assert np.isscalar(chi_scalar), "chi should return scalar for scalar input"
        assert np.isscalar(dydm_scalar), "dydm should return scalar for scalar input"
        assert isinstance(chi_hat_scalar, (int, float, np.number)), (
            "chi_hat should return numeric scalar"
        )
        assert isinstance(chi_scalar, (int, float, np.number)), (
            "chi should return numeric scalar"
        )
        assert isinstance(dydm_scalar, (int, float, np.number)), (
            "dydm should return numeric scalar"
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

            chi_hat_array = camber_instance.chi_hat(m_array)
            chi_array = camber_instance.chi(m_array)
            dydm_array = camber_instance.dydm(m_array)

            assert chi_hat_array.shape == m_array.shape, (
                f"chi_hat shape mismatch for {shape}"
            )
            assert chi_array.shape == m_array.shape, f"chi shape mismatch for {shape}"
            assert dydm_array.shape == m_array.shape, f"dydm shape mismatch for {shape}"

            assert isinstance(chi_hat_array, np.ndarray), (
                f"chi_hat should return ndarray for shape {shape}"
            )
            assert isinstance(chi_array, np.ndarray), (
                f"chi should return ndarray for shape {shape}"
            )
            assert isinstance(dydm_array, np.ndarray), (
                f"dydm should return ndarray for shape {shape}"
            )
