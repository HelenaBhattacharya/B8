import numpy as np
from bragg_engine.mapping import compute_energy_map, rotation_matrix
from bragg_engine.solid_angle import compute_solid_angle
from config import CCD_SHAPE, PIXEL_SIZE

def remove_high_ADU_pixels(adu_weighted_ccd, ccd_redistributed, high_adu_positions):
    """
    Zeroes out pixels around high-ADU positions in both CCD maps to prevent double-counting.

    Args:
        adu_weighted_ccd (np.ndarray): SPC ADU-weighted CCD map.
        ccd_redistributed (np.ndarray): High-ADU redistributed CCD map.
        high_adu_positions (list or np.ndarray): (y, x) coordinates of bright pixels to mask.
    """

    initial_nonzero_adu = np.count_nonzero(adu_weighted_ccd)
    initial_nonzero_redistributed = np.count_nonzero(ccd_redistributed)

    for y, x in high_adu_positions:
        for dy in range(-2, 2):
            for dx in range(-2, 2):
                yy, xx = y + dy, x + dx
                height, width = CCD_SHAPE
                if 0 <= yy < height and 0 <= xx < width:
                    adu_weighted_ccd[yy, xx] = 0
                    ccd_redistributed[yy, xx] = 0

    final_nonzero_adu = np.count_nonzero(adu_weighted_ccd)
    final_nonzero_redistributed = np.count_nonzero(ccd_redistributed)

    print(f"Removed high-ADU pixels:")
    print(f"  ADU-weighted CCD: {initial_nonzero_adu - final_nonzero_adu} pixels set to 0.")
    print(f"  Redistributed CCD: {initial_nonzero_redistributed - final_nonzero_redistributed} pixels set to 0.")

def extract_photon_hits(cluster_pixel_map, adu_weighted_ccd, ccd_redistributed):
    """
    Extract photon hits separately from SPC and high-ADU data.

    Returns:
        spc_hits (np.ndarray): SPC photon hits as [(y, x, ADU), ...].
        high_adu_hits (np.ndarray): High-ADU photon hits as [(y, x, ADU), ...].
    """
    y_spc, x_spc = np.where(cluster_pixel_map > 0)
    adu_spc = adu_weighted_ccd[y_spc, x_spc]

    y_adu, x_adu = np.where(ccd_redistributed > 0)
    adu_adu = ccd_redistributed[y_adu, x_adu]

    spc_hits = np.column_stack((y_spc, x_spc, adu_spc))
    high_adu_hits = np.column_stack((y_adu, x_adu, adu_adu))

    return spc_hits, high_adu_hits


import numpy as np

def resolve_overlaps(spc_hits, high_adu_hits):
    """
    Resolves overlapping pixels between SPC and high-ADU hits.
    Keeps only the high-ADU hits if overlap is detected.

    Args:
        spc_hits (np.ndarray): Array of SPC hits [y, x, adu].
        high_adu_hits (np.ndarray): Array of high-ADU hits [y, x, adu].

    Returns:
        tuple: (clean_spc_hits, clean_high_adu_hits, num_removed_overlaps)
    """
    # Find overlapping pixels based on (y, x) positions
    spc_coords = set(tuple(hit[:2]) for hit in spc_hits)
    high_adu_coords = set(tuple(hit[:2]) for hit in high_adu_hits)

    overlapping_coords = spc_coords & high_adu_coords
    num_removed_overlaps = len(overlapping_coords)

    # Debug printing to confirm overlap correctness:
    if num_removed_overlaps > 0:
        print(f"[DEBUG] Found {num_removed_overlaps} overlaps. Example coordinates:")
        for coord in list(overlapping_coords)[:5]:  # limit print to first 5 overlaps
            print(f"  Overlap at pixel: {coord}")

    # Filter out overlapping SPC hits
    clean_spc_hits = np.array([hit for hit in spc_hits if tuple(hit[:2]) not in overlapping_coords])
    clean_high_adu_hits = high_adu_hits  # High-ADU hits are kept entirely

    return clean_spc_hits, clean_high_adu_hits, num_removed_overlaps

def map_photon_energies(photon_hits, optimized_params):
    """
    Maps photon hits to energies using energy calibration.

    Args:
        photon_hits (np.ndarray): Photon hits [(y, x, ADU), ...].
        optimized_params (tuple): Calibration parameters.

    Returns:
        photon_energies (np.ndarray): Photon energies (eV).
        photon_adus (np.ndarray): Corresponding ADU values.
    """
    E_ij, _, _ = compute_energy_map(optimized_params)

    y_coords = photon_hits[:, 0].astype(int)
    x_coords = photon_hits[:, 1].astype(int)
    photon_adus = photon_hits[:, 2]

    photon_energies = E_ij[y_coords, x_coords]

    return photon_energies, photon_adus


def sum_photon_hits(photon_energies, photon_adus, energy_bins):
    """
    Builds ADU-weighted photon energy histogram.

    Returns:
        hist_counts (np.ndarray): Photon counts per energy bin.
    """
    weighted_energies = np.repeat(photon_energies, photon_adus.astype(int))
    hist_counts, _ = np.histogram(weighted_energies, bins=energy_bins)

    return hist_counts


def compute_solid_angle_map(optimized_params):
    """
    Computes solid-angle map for the CCD.

    Args:
        optimized_params (tuple): Calibration parameters.

    Returns:
        Omega_ij (np.ndarray): Solid angle per CCD pixel.
    """
    D, theta_bar_B, alpha_x, alpha_y = optimized_params
    R = rotation_matrix(np.radians(alpha_x), np.radians(alpha_y), 0)
    _, x_prime, y_prime = compute_energy_map(optimized_params)

    Omega_ij = compute_solid_angle(x_prime, y_prime, D, theta_bar_B, R)

    return Omega_ij

def simple_solid_angle_per_bin(E_ij, Omega_ij, energy_bins):
    Omega_E = np.zeros(len(energy_bins) - 1)
    for i, (e_min, e_max) in enumerate(zip(energy_bins[:-1], energy_bins[1:])):
        mask = (E_ij >= e_min) & (E_ij < e_max)
        Omega_E[i] = np.sum(Omega_ij[mask])
    return Omega_E

def solid_angle_correction(hist_counts, energy_bins, E_ij, Omega_ij):
    """
    Applies solid-angle correction to energy histogram using fractional binning.

    Args:
        hist_counts (np.ndarray): Photon counts per energy bin.
        energy_bins (np.ndarray): Energy bin edges.
        E_ij (np.ndarray): Energy map per pixel.
        Omega_ij (np.ndarray): Solid-angle map per pixel.

    Returns:
        corrected_intensity (np.ndarray): Solid-angle corrected photon counts.
    """
    Omega_E = simple_solid_angle_per_bin(E_ij, Omega_ij, energy_bins)
    corrected_intensity = hist_counts / Omega_E
    corrected_intensity[np.isnan(corrected_intensity)] = 0  # Avoid NaNs or Infs

    return corrected_intensity

# # Helper function (placed here if needed)
# def rotation_matrix(alpha_x, alpha_y, alpha_z):
#     """
#     Creates a rotation matrix from three rotation angles (in radians).
#     """
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(alpha_x), -np.sin(alpha_x)],
#         [0, np.sin(alpha_x), np.cos(alpha_x)]
#     ])
#
#     Ry = np.array([
#         [np.cos(alpha_y), 0, np.sin(alpha_y)],
#         [0, 1, 0],
#         [-np.sin(alpha_y), 0, np.cos(alpha_y)]
#     ])
#
#     Rz = np.array([
#         [np.cos(alpha_z), -np.sin(alpha_z), 0],
#         [np.sin(alpha_z), np.cos(alpha_z), 0],
#         [0, 0, 1]
#     ])
#
#     return Rz @ Ry @ Rx