import os
print("Current Working Directory:", os.getcwd())  # This shows where Python is looking for the file
import numpy as np
from bragg_engine.mapping import compute_energy_map
from bragg_engine.optimisation import optimize_parameters
from bragg_engine.curve_fitting import fit_quadratic_curve
from config import LATTICE_SPACING, PIXEL_SIZE

# Constants
h = 6.62607015e-34  # Planck's constant (J·s)
c = 2.99792458e8  # Speed of light (m/s)
joule_to_ev = 1.60218e-19  # Joules to eV conversion
d = LATTICE_SPACING  # Lattice spacing (m)
sqrt_2ln2_factor = 2 * np.sqrt(2 * np.log(2))  # Convert sigma to FWHM

def extract_optimized_params():
    """
    Extracts D (source-CCD distance) and θ̄_B (Bragg angle) from optimized_params.npy.
    If the file does not exist, runs optimization first.

    Returns:
        tuple: (D, theta_bar_B, alpha_x, alpha_y)
    """
    if not os.path.exists("optimized_params.npy"):
        print("\n Missing optimized_params.npy. Running optimization...")
        optimized_params = optimize_parameters()
        np.save("optimized_params.npy", np.array(optimized_params, dtype=np.float64))
    else:
        optimized_params = np.load("optimized_params.npy", allow_pickle=True)

    if len(optimized_params) < 4:
        raise ValueError("Error: optimized_params.npy is invalid! Expected at least 4 values.")

    D = optimized_params[0]  # Source-CCD distance (meters)
    theta_bar_B = optimized_params[1]  # Bragg angle (degrees)
    alpha_x = optimized_params[2]
    alpha_y = optimized_params[3]

    return D, theta_bar_B, alpha_x, alpha_y

def compute_energy_dispersion(E_ij, x_prime, energy_level):
    """
    Computes dE/dx at the closest energy contour.

    Args:
        E_ij (np.ndarray): Energy map.
        x_prime (np.ndarray): X-coordinates (meters).
        energy_level (float): Reference energy (eV).

    Returns:
        float: Mean dE/dx (eV/m).
    """
    # **Find the closest energy index in E_ij**
    energy_differences = np.abs(E_ij - energy_level)
    energy_index = np.unravel_index(np.argmin(energy_differences, axis=None), E_ij.shape)

    # **Extract column at the found energy index**
    energy_column = E_ij[:, energy_index[1]]  # Extracts energy values along the vertical axis
    x_column = x_prime[:, energy_index[1]]  # Extracts x values along the same column

    dE_dx = np.gradient(energy_column, PIXEL_SIZE)

    # **Ensure valid dE_dx values
    if np.any(np.isnan(dE_dx)) or np.any(np.isinf(dE_dx)):
        raise ValueError(f"Error: Computed dE/dx contains NaN or Inf values at {energy_level} eV!")

    return np.mean(dE_dx)


def calc_pixel_broadening(theta_B_deg, D):
    """Computes broadening due to finite pixel size."""
    theta_B_rad = np.deg2rad(theta_B_deg)
    dE_dx = (h * c) / (2 * d * joule_to_ev) * np.abs(np.cos(theta_B_rad) / (D * np.sin(theta_B_rad) ** 2))
    return dE_dx * PIXEL_SIZE


def calc_source_broadening(theta_B_deg, D, spot_area_m2):
    """Computes broadening due to finite source size (modeled as a circular beam)."""
    theta_B_rad = np.deg2rad(theta_B_deg)
    effective_source_radius = np.sqrt(spot_area_m2 / np.pi)  # Convert area to radius
    dE_dx = (h * c) / (2 * d * joule_to_ev) * np.abs(np.cos(theta_B_rad) / (D * np.sin(theta_B_rad) ** 2))
    return dE_dx * effective_source_radius


def calc_rocking_curve_broadening(sigma_pixels, dE_dx_pixel):
    """
    Computes broadening due to the rocking curve width using fitted σ from curve_fitting.

    Args:
        sigma_pixels (float): Mean fitted Gaussian width (σ) in pixels.
        dE_dx_pixel (float): Energy dispersion (eV/m).

    Returns:
        float: Rocking curve broadening in eV.
    """
    fwhm_meters = sqrt_2ln2_factor * sigma_pixels * PIXEL_SIZE  # Convert σ to FWHM
    return dE_dx_pixel * fwhm_meters


def calc_total_energy_resolution(theta_B_deg, D, sigma_pixels, dE_dx_pixel, spot_area_m2):
    """Computes total energy resolution using quadrature sum."""
    ΔE_pixel = calc_pixel_broadening(theta_B_deg, D)
    ΔE_source = calc_source_broadening(theta_B_deg, D, spot_area_m2)
    ΔE_rocking = calc_rocking_curve_broadening(sigma_pixels, dE_dx_pixel)

    ΔE_total = np.sqrt(ΔE_pixel ** 2 + ΔE_source ** 2 + ΔE_rocking ** 2)

    return {
        'pixel_broadening': ΔE_pixel,
        'source_broadening': ΔE_source,
        'rocking_curve_broadening': ΔE_rocking,
        'total_broadening': ΔE_total
    }