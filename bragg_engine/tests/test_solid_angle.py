import numpy as np
import pytest
import matplotlib.pyplot as plt
from bragg_engine.solid_angle import compute_solid_angle
from bragg_engine.mapping import rotation_matrix

# Constants for testing
D = 0.083  # Example distance (meters)
theta_bar_B = 37.2
R_identity = rotation_matrix(0, 0, 0)

@pytest.fixture
def mock_coordinates():
    x_prime, y_prime = np.meshgrid(
        np.linspace(-1e-3, 1e-3, 50),
        np.linspace(-1e-3, 1e-3, 50)
    )
    return x_prime, y_prime

def test_compute_solid_angle_shape(mock_coordinates):
    x_prime, y_prime = mock_coordinates
    Omega = compute_solid_angle(x_prime, y_prime, D, theta_bar_B, R_identity)
    assert Omega.shape == x_prime.shape, "Shape mismatch."

def test_compute_solid_angle_positive(mock_coordinates):
    x_prime, y_prime = mock_coordinates
    Omega = compute_solid_angle(x_prime, y_prime, D, theta_bar_B, R_identity)
    assert np.all(Omega > 0), "Non-positive solid angle."

def test_compute_solid_angle_rotation_effect(mock_coordinates):
    x_prime, y_prime = mock_coordinates
    Omega_no_rotation = compute_solid_angle(x_prime, y_prime, D, theta_bar_B, R_identity)
    R_rotated = rotation_matrix(5, 5, 0)
    Omega_rotated = compute_solid_angle(x_prime, y_prime, D, theta_bar_B, R_rotated)
    assert not np.allclose(Omega_no_rotation, Omega_rotated), "Rotation has no effect."

def test_compute_solid_angle_invalid_input():
    with pytest.raises(ValueError):
        compute_solid_angle(np.array([[0, 1]]), np.array([[0, 1, 2]]), D, theta_bar_B, R_identity)
