import numpy as np
from config import CCD_SHAPE, PIXEL_SIZE, LATTICE_SPACING, ALPHA_Z

# Constants
h = 6.62607015e-34  # Planck's constant (J·s)
c = 2.99792458e8  # Speed of light (m/s)
d = LATTICE_SPACING  # Lattice spacing (m)
pixel_size = PIXEL_SIZE # Pixel size in meters
ccd_size = CCD_SHAPE # CCD dimensions (pixels)
joule_to_ev = 1.60218e-19  # Conversion factor

def rotation_matrix(alpha_x, alpha_y, alpha_z=0):
    """Computes the full rotation matrix from Euler angles."""
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha_x), -np.sin(alpha_x)], [0, np.sin(alpha_x), np.cos(alpha_x)]])
    Ry = np.array([[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
    Rz = np.array([[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def compute_energy_map(params):
    """Computes the energy map E_ij with corrected coordinate mapping."""
    D, theta_bar_B, alpha_x, alpha_y = params
    alpha_z = ALPHA_Z # Keep α_z = 0 for now
    theta_bar_B = np.radians(theta_bar_B)

    # Compute rotation matrix
    R = rotation_matrix(np.radians(alpha_x), np.radians(alpha_y), np.radians(alpha_z))

    # Compute central ray r0
    r0 = np.array([D * np.cos(theta_bar_B), 0, D * np.sin(theta_bar_B)])

    # Generate CCD coordinate grid (Using Centered Indexing like Old Code)
    height, width = CCD_SHAPE
    x_range = np.linspace(-width // 2, width // 2, width)
    y_range = np.linspace(-height // 2, height // 2, height)
    x_grid, y_grid = np.meshgrid(x_range, y_range, indexing='ij')

    # **Rotate CCD by 90 degrees clockwise** (Matching Old Implementation)
    x_grid_rotated = np.rot90(x_grid, k=-1)
    y_grid_rotated = np.rot90(y_grid, k=-1)

    # Convert to real-world coordinates (Matching the Old Implementation)
    x_prime = -x_grid_rotated * pixel_size  #  Flip X to match physical space
    y_prime = y_grid_rotated * pixel_size  # No inversion

    # Define r_ij' relative to the CCD center
    r_ij_prime = np.stack([-x_prime, y_prime, np.zeros_like(x_prime)], axis=-1)

    # Apply rotation
    r_ij = np.einsum('ij,xyj->xyi', R, r_ij_prime)

    # Compute r_ray
    r_ray = r0[None, None, :] + r_ij

    # Normalize to get r_ray_hat
    r_ray_norm = np.linalg.norm(r_ray, axis=-1, keepdims=True)
    r_ray_hat = r_ray / r_ray_norm

    # Compute E_ij using z_hat dot r_ray_hat = hc / 2dE
    z_hat = np.array([0, 0, 1])
    numerator = h * c / (2 * d)
    denominator = np.dot(r_ray_hat, z_hat)

    # Compute E_ij in eV
    E_ij = (numerator / denominator) / joule_to_ev
    E_ij[denominator == 0] = np.nan

    return E_ij, x_prime, y_prime