import numpy as np
from config import PIXEL_SIZE
from bragg_engine.mapping import rotation_matrix

def compute_solid_angle(x_prime, y_prime, D, theta_bar_B, R):
    """
    Compute the solid angle contribution for each pixel.

    Args:
        x_prime (np.ndarray): X-coordinates of CCD pixels (meters).
        y_prime (np.ndarray): Y-coordinates of CCD pixels (meters).
        D (float): Crystal-to-detector distance.
        theta_bar_B (float): Bragg angle (degrees).
        R (np.ndarray): 3×3 rotation matrix.

    Returns:
        np.ndarray: Solid angle per pixel.
    """
    theta_bar_B = np.radians(theta_bar_B)  # Convert to radians
    z_prime = np.dot(R, np.array([0, 0, 1]))  # Apply rotation to lab z-axis
    r0 = D * np.array([np.cos(theta_bar_B), 0, np.sin(theta_bar_B)])
    D_prime = np.dot(z_prime, r0)
    x0 = D_prime * z_prime
    x0_prime = r0 - x0

    # Convert 2D x_prime, y_prime into 3D vectors
    r_ij_prime = np.stack([-x_prime, y_prime, np.zeros_like(x_prime)], axis=-1)

    # Apply inverse rotation to each pixel coordinate
    x_double_prime = r_ij_prime + np.einsum('ij,j->i', np.linalg.inv(R), x0_prime)

    # Extract transformed coordinates
    x_double = x_double_prime[..., 0]
    y_double = x_double_prime[..., 1]

    # Compute solid angle per pixel
    Omega = (D_prime * PIXEL_SIZE ** 2) / ((x_double ** 2 + y_double ** 2 + D_prime ** 2) ** (3 / 2))

    return Omega
