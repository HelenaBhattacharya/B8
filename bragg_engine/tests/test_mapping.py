import numpy as np
import pytest
from bragg_engine.mapping import compute_energy_map, rotation_matrix
from config import PIXEL_SIZE, CCD_SHAPE

@pytest.fixture
def default_params():
    D = 0.5  # Distance (m)
    theta_bar_B = 75  # Bragg angle (degrees)
    alpha_x = 0  # Rotation about X (degrees)
    alpha_y = 0  # Rotation about Y (degrees)
    return D, theta_bar_B, alpha_x, alpha_y


def test_rotation_matrix_identity():
    """Rotation matrix with zero angles should yield identity matrix."""
    R_expected = np.eye(3)
    R_computed = rotation_matrix(0, 0, 0)
    assert np.allclose(R_computed, R_expected), "Identity rotation matrix failed."


def test_energy_map_shape(default_params):
    """Check if E_ij has the correct CCD dimensions."""
    E_ij, x_prime, y_prime = compute_energy_map(default_params)
    assert E_ij.shape == CCD_SHAPE, f"Energy map shape mismatch: expected {CCD_SHAPE}, got {E_ij.shape}."
    assert x_prime.shape == CCD_SHAPE, f"x_prime shape mismatch."
    assert y_prime.shape == CCD_SHAPE, f"y_prime shape mismatch."


def test_energy_map_values_center(default_params):
    """Check if the central pixel energy matches the expected theoretical value."""
    E_ij, _, _ = compute_energy_map(default_params)

    # Central pixel index
    center = (CCD_SHAPE[0] // 2, CCD_SHAPE[1] // 2)

    # Theoretical energy at central pixel based on input params (Bragg condition)
    h = 6.62607015e-34  # Planck's constant
    c = 2.99792458e8  # Speed of light
    d = 7.98e-10  # Lattice spacing
    joule_to_ev = 1.60218e-19

    E_expected = (h * c) / (2 * d * np.sin(np.radians(default_params[1])))
    E_expected /= joule_to_ev

    E_center = E_ij[center]
    assert np.isclose(E_center, E_expected, atol=0.1), f"Center energy mismatch: {E_center} vs {E_expected}"


@pytest.mark.parametrize("angles", [(0, 0), (1, 0), (0, 1), (2, -2), (-5, 3)])
def test_energy_map_rotation_effect(angles, default_params):
    D, theta_bar_B, _, _ = default_params
    alpha_x, alpha_y = angles

    params_no_rotation = (D, theta_bar_B, 0, 0)
    params_with_rotation = (D, theta_bar_B, alpha_x, alpha_y)

    E_no_rotation, _, _ = compute_energy_map(params_no_rotation)
    E_with_rotation, _, _ = compute_energy_map(params_with_rotation)

    if angles == (0, 0):
        assert np.allclose(E_no_rotation, E_with_rotation), "Energy map should be identical for zero rotations."
    else:
        assert not np.allclose(E_no_rotation, E_with_rotation), f"Rotation angles {angles} did not alter energy map as expected."

def test_extreme_angles_handling():
    """Check energy calculation with extreme angles to verify stability."""
    extreme_params = (0.5, 89.9, 10, -10)  # Bragg angle approaching 90 degrees
    E_ij, _, _ = compute_energy_map(extreme_params)

    assert np.all(np.isfinite(E_ij[~np.isnan(E_ij)])), "Non-finite values detected in energy map for extreme angles."