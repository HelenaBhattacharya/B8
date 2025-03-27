import numpy as np
import matplotlib.pyplot as plt
from spectral_reconstruction.spectral_processing import (
    extract_photon_hits,
    resolve_overlaps,
    map_photon_energies,
    sum_photon_hits,
    compute_solid_angle_map,
    solid_angle_correction
)
from bragg_engine.load import get_ccd_image
from spc_engine.pedestal import compute_pedestal
import os
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
from bragg_engine.load import get_ccd_image
from bragg_engine.mapping import compute_energy_map
import os
from config import HDF5_FILE, OPTIMIZED_PARAMS_FILE, CCD_SHAPE


# Explicit linear energy mapping helper function
def simple_linear_energy_mapping(columns, col_ref1=1284, energy_ref1=1188.0, col_ref2=1424, energy_ref2=1218.5):
    a = (energy_ref2 - energy_ref1) / (col_ref2 - col_ref1)
    b = energy_ref1 - a * col_ref1
    return a * columns + b

def visual_test_simple_0th_order_spectrum(file_name=str(HDF5_FILE), image_index=8, start_column=50):
    """
    Corrected 0th-order spectrum: explicitly mapped and reversed energy axis.
    """
    # Define paths clearly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, '..', '..'))
    file_path = str(HDF5_FILE)

    # Load raw CCD data explicitly
    ccd_image = get_ccd_image(file_path, image_index)

    # Perform simple vertical integration (no pedestal or thresholding)
    spectrum = np.sum(ccd_image[:, start_column:], axis=0)
    columns = np.arange(start_column, ccd_image.shape[1])

    # Energy mapping explicitly based on your provided pixel-energy references:
    # Pixel 1284 → 1218.5 eV, Pixel 1424 → 1188 eV
    col_ref1, energy_ref1 = 1284, 1218.5
    col_ref2, energy_ref2 = 1424, 1188.0

    # Linear calibration: Energy = a*column + b
    a = (energy_ref2 - energy_ref1) / (col_ref2 - col_ref1)
    b = energy_ref1 - a * col_ref1

    # Map columns explicitly to energies
    energies = a * columns + b

    # Reverse arrays explicitly to ensure increasing energy left to right
    energies = energies[::-1]
    spectrum = spectrum[::-1]

    # Plot explicitly and clearly labeled
    plt.figure(figsize=(10, 6))
    plt.plot(energies, spectrum, color='purple', linewidth=0.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Integrated ADU Counts (Vertical Sum)')
    plt.title(f'Simple 0th-order Spectrum (Image {image_index}) - Correct Energy Mapping')
    plt.grid(True)

    # Explicit vertical lines for known reference points
    plt.axvline(1188.0, color='red', linestyle='--', label='Ge Lα (1188 eV)')
    plt.axvline(1218.5, color='green', linestyle='--', label='Ge Lβ (1218.5 eV)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visual_test_simple_1st_order_spectrum(file_name=str(HDF5_FILE), optimized_params=None, image_index=8, start_column=50, ADU_threshold=80):
    """
    Simple 1st-order spectrum with accurate energy mapping from main code and simple threshold.
    """
    # Paths setup explicitly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, '..', '..'))
    file_path = str(HDF5_FILE)

    ccd_image = get_ccd_image(file_path, image_index)

    # Apply simple threshold directly (no pedestal subtraction)
    thresholded_image = np.copy(ccd_image)
    thresholded_image[thresholded_image < ADU_threshold] = 0

    # Vertical integration after thresholding
    spectrum_counts = np.sum(thresholded_image[:, start_column:], axis=0)

    # Accurate energy calibration (exactly as in main spectral reconstruction)
    E_map, _, _ = compute_energy_map(optimized_params)
    energies = E_map[E_map.shape[0]//2, start_column:]

    plt.figure(figsize=(10, 6))
    plt.plot(energies, spectrum_counts, color='blue', linewidth=0.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Integrated Photon Counts (Thresholded ADU)')
    plt.title(f'1st-order Spectrum (Image {image_index}), Threshold {ADU_threshold} ADU')
    plt.grid(True)

    plt.axvline(1188.0, color='red', linestyle='--', label='Ge Lα (1188 eV)')
    plt.axvline(1218.5, color='green', linestyle='--', label='Ge Lβ (1218.5 eV)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visual_test_2nd_order_spectrum(file_name=str(HDF5_FILE), optimized_params=None, image_index=8):
    """
    2nd-order spectrum: Uses SPC and high-ADU data with accurate energy mapping.
    """

    # Paths setup explicitly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, '..', '..'))
    file_path = str(HDF5_FILE)

    # Load explicitly stored SPC engine output files for image 8
    folder = os.path.join(project_root, str(image_index))
    cluster_map_file = os.path.join(folder, "cluster_pixel_map.npy")
    adu_weighted_file = os.path.join(folder, "adu_weighted_ccd_final.npy")
    redistributed_file = os.path.join(folder, "ccd_redistributed.npy")

    cluster_pixel_map = np.load(cluster_map_file)
    adu_weighted_ccd = np.load(adu_weighted_file)
    ccd_redistributed = np.load(redistributed_file)

    # Extract photon hits explicitly from loaded SPC data
    spc_hits, high_adu_hits = extract_photon_hits(cluster_pixel_map, adu_weighted_ccd, ccd_redistributed)

    # Resolve overlaps explicitly (important for correctness)
    clean_spc_hits, clean_high_adu_hits, overlaps_removed = resolve_overlaps(spc_hits, high_adu_hits)

    # Combine cleaned hits explicitly
    combined_hits = np.vstack([clean_spc_hits, clean_high_adu_hits])

    # Map photon hits explicitly to energies
    photon_energies, photon_adus = map_photon_energies(combined_hits, optimized_params)

    # Define energy bins clearly
    ENERGY_MIN, ENERGY_MAX, BIN_WIDTH = 1100, 1600, 1
    ENERGY_BINS = np.arange(ENERGY_MIN, ENERGY_MAX + BIN_WIDTH, BIN_WIDTH)

    # Compute energy histogram explicitly
    hist_counts = sum_photon_hits(photon_energies, photon_adus, ENERGY_BINS)

    # Compute solid angle correction explicitly (consistent with main code)
    E_map, _, _ = compute_energy_map(optimized_params)
    Omega_ij = compute_solid_angle_map(optimized_params)

    corrected_counts = solid_angle_correction(hist_counts, ENERGY_BINS, E_map, Omega_ij)

    bin_centers = (ENERGY_BINS[:-1] + ENERGY_BINS[1:]) / 2

    # Plot the explicitly solid-angle corrected spectrum clearly
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, corrected_counts, color='red', linewidth=0.7)  # plot for smoothness
    plt.xlabel('Energy (eV)')
    plt.ylabel('Solid-angle Corrected Photon Counts')
    plt.title(f'2nd-order Spectrum (SPC & high-ADU, Image {image_index} Only)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Highlight known spectral lines explicitly
    # plt.axvline(1188.0, color='red', linestyle='--', label='Ge Lα (1188 eV)')
    # plt.axvline(1218.5, color='green', linestyle='--', label='Ge Lβ (1218.5 eV)')
    # plt.legend()

    plt.tight_layout()
    plt.show()

    # Explicit summary print statement for clarity
    print(f"Image {image_index} SPC photon hits (cleaned): {len(clean_spc_hits)}")
    print(f"Image {image_index} high-ADU photon hits: {len(clean_high_adu_hits)}")
    print(f"Overlapping hits removed: {overlaps_removed}")
    print(f"Total photon hits used for spectrum: {len(combined_hits)}")


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def compute_pedestal(raw_ccd_image):
    pixel_values = raw_ccd_image.flatten()
    bins = np.arange(0, np.max(pixel_values), 2)
    hist, bin_edges = np.histogram(pixel_values, bins=bins)
    A_init = np.max(hist)
    mu_init = bin_edges[np.argmax(hist)]
    sigma_init = 10

    fit_range = (bin_edges[:-1] >= 0) & (bin_edges[:-1] <= 100)
    x_fit = bin_edges[:-1][fit_range]
    y_fit = hist[fit_range]

    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[A_init, mu_init, sigma_init])
        _, mu_fit, sigma_fit = popt
    except RuntimeError:
        mu_fit, sigma_fit = mu_init, sigma_init

    return mu_fit, sigma_fit

def visual_test_photon_energy_histogram():
    photon_energies = np.random.normal(1200, 10, 1000)
    photon_adus = np.random.randint(1, 5, 1000)
    energy_bins = np.linspace(1150, 1250, 100)
    raw_counts = sum_photon_hits(photon_energies, photon_adus, energy_bins)
    optimized_params = (0.083, 37.2, 0.0, 0.0)
    E_ij = np.linspace(1150, 1250, CCD_SHAPE[0] * CCD_SHAPE[1]).reshape(CCD_SHAPE)
    Omega_ij = np.random.uniform(1e-6, 1e-5, E_ij.shape)
    corrected_counts = solid_angle_correction(raw_counts, energy_bins, E_ij, Omega_ij)

    plt.figure(figsize=(10, 6))
    plt.step(energy_bins[:-1], raw_counts, label="Raw Counts")
    plt.step(energy_bins[:-1], corrected_counts, label="Solid-Angle Corrected")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Counts")
    plt.title("Photon Energy Histogram")
    plt.legend()
    plt.grid(True)
    plt.show()

def visual_test_overlap_resolution():
    spc_hits = np.array([[100, 100, 120], [101, 101, 115], [150, 150, 110]])
    high_adu_hits = np.array([[100, 100, 250], [120, 120, 260]])
    clean_spc, clean_high_adu, removed = resolve_overlaps(spc_hits, high_adu_hits)
    print(f"Overlapping pixels removed: {removed}")

    plt.figure(figsize=(8, 8))
    plt.scatter(spc_hits[:,1], spc_hits[:,0], c='blue', marker='x', label='SPC original')
    plt.scatter(high_adu_hits[:,1], high_adu_hits[:,0], c='red', marker='o', label='High-ADU Hits')
    plt.scatter(clean_spc[:,1], clean_spc[:,0], c='cyan', marker='x', label='SPC cleaned')

    plt.gca().invert_yaxis()
    plt.title("Overlap Resolution Visualization")
    plt.xlabel("CCD Pixel X")
    plt.ylabel("CCD Pixel Y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = str(HDF5_FILE)
    optimized_params = np.load(str(OPTIMIZED_PARAMS_FILE), allow_pickle=True)

    print("Running Visual Test 1: Photon Energy Histogram...")
    visual_test_photon_energy_histogram()

    print("\nRunning Visual Test 2: Overlap Resolution...")
    visual_test_overlap_resolution()

    print("Running visual test: Corrected 0th-order Spectrum...")
    visual_test_simple_0th_order_spectrum(file_path, image_index=8, start_column=50)

    print("\nRunning visual test: 1st-order Spectrum with accurate mapping and simple thresholding...")
    visual_test_simple_1st_order_spectrum(file_path, optimized_params, image_index=8, start_column=50, ADU_threshold=80)

    visual_test_2nd_order_spectrum(file_path, optimized_params, image_index=8)