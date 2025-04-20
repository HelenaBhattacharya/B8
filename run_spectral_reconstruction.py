"""
run_spectral_reconstruction.py

Performs spectral reconstruction from CCD photon hit data using Bragg-mapped energy bins.
Combines SPC and high-ADU events across selected images, applies solid angle correction,
and produces the final intensity spectrum with propagated Poisson errors.

Outputs:
- Photon hit statistics and overlap removal
- Energy-resolved photon counts with solid angle correction
- Visualizations of CCD hits, spectrum, and intensity

Usage:
    python spectral_reconstruction.py [file_name] [--images 0 1 2 ...]
"""

import argparse
import numpy as np
import os

# Core processing and mapping
from bragg_engine.preprocessing import sum_all_ccd_images
from bragg_engine.mapping import compute_energy_map
from spc_engine.intensity_error import poisson_error

# Photon extraction and reconstruction pipeline
from spectral_reconstruction.spectral_processing import (
    remove_high_ADU_pixels,
    extract_photon_hits,
    resolve_overlaps,
    map_photon_energies,
    compute_solid_angle_map,
    solid_angle_correction,
    simple_solid_angle_per_bin
)

# Spectrum visualizations
from spectral_reconstruction.visualisation import (
    plot_photon_counts_vs_energy,
    plot_solid_angle_adjusted_spectrum,
    plot_intensity_vs_energy,
    plot_log_intensity_vs_energy,
    plot_normalized_spectrum,
    plot_ccd_hits,
    plot_extracted_spectrum_lineout,
    plot_solid_angle_vs_energy
)

# Global parameters
from config import (
    HDF5_FILE,
    CCD_SHAPE,
    OPTIMIZED_PARAMS_FILE,
    ENERGY_MIN,
    ENERGY_MAX,
    BIN_WIDTH,
    PIXEL_SIZE,
    HIGH_ADU_CLUSTER_THRESHOLDS,
    SPC_SIGMA_N,
    SPC_ADU_SINGLE_PHOTON
)

# Energy bin edges for histogramming
ENERGY_BINS = np.arange(ENERGY_MIN, ENERGY_MAX + BIN_WIDTH, BIN_WIDTH)


def run_spectral_reconstruction(filename=HDF5_FILE, image_indices=None):
    """
    Main spectral reconstruction routine.

    Combines photon hit data across selected CCD images, resolves SPC and high-ADU overlaps,
    applies solid angle correction, and produces energy-resolved spectra.

    Steps:
        - Load energy and solid angle maps using optimized Bragg parameters
        - Extract and classify photon hits from SPC and high-ADU sources
        - Resolve spatial overlaps between SPC and high-ADU clusters
        - Assign photon counts per cluster using ADU thresholds
        - Map hit positions to energy and compute histograms
        - Apply solid angle correction and propagate Poisson errors
        - Generate final diagnostic plots and summary statistics

    Parameters:
        filename (str): Path to HDF5 file containing CCD data.
        image_indices (list[int] or None): List of image indices to process.
                                           If None, processes all 20 images.
    """
    print(f"Running spectral reconstruction for file: {filename}")

    # Load mapping parameters and full CCD image sum
    optimized_params = np.load(OPTIMIZED_PARAMS_FILE, allow_pickle=True)
    summed_image, num_high_adu, high_adu_positions = sum_all_ccd_images(filename)
    print(f"High-ADU pixels identified: {num_high_adu}")

    # Image indices to process
    images_to_process = image_indices if image_indices is not None else range(20)

    # Initialize accumulators
    total_removed_overlaps = 0
    total_spc_photons_before_overlap = 0
    total_spc_photons_after_overlap = 0
    total_high_adu_photons = 0
    x_hits_all, y_hits_all = [], []

    accumulated_corrected_counts = np.zeros(len(ENERGY_BINS) - 1)
    accumulated_errors_squared = np.zeros(len(ENERGY_BINS) - 1)

    # Compute energy and solid angle maps
    E_ij, _, _ = compute_energy_map(optimized_params)
    Omega_ij = compute_solid_angle_map(optimized_params)
    Omega_E = simple_solid_angle_per_bin(E_ij, Omega_ij, ENERGY_BINS)

    # Assign number of photons per hit based on ADU
    def assign_photon_counts(hits_array, high_adu=False):
        """
        Assigns photon count estimates to each cluster based on ADU.

        Parameters:
            hits_array (ndarray): Array of detected photon hits [x, y, adu].
            high_adu (bool): Whether to apply multi-photon thresholds.

        Returns:
            np.ndarray: Array of [x, y, photon_count] for valid hits.
        """
        photon_counts = []
        T_secondary = SPC_ADU_SINGLE_PHOTON - SPC_SIGMA_N

        for hit in hits_array:
            x, y, adu = hit
            if not high_adu:
                if T_secondary <= adu < HIGH_ADU_CLUSTER_THRESHOLDS[1]:
                    photons = 1
                else:
                    photons = 0
            else:
                if adu < HIGH_ADU_CLUSTER_THRESHOLDS[2]:
                    photons = 2
                elif adu < HIGH_ADU_CLUSTER_THRESHOLDS[3]:
                    photons = 3
                else:
                    photons = 4
            if photons > 0:
                photon_counts.append([x, y, photons])
        return np.array(photon_counts)

    # --- Loop over images ---
    for idx in images_to_process:
        print(f"\nProcessing image {idx}...")

        folder = str(idx)
        files = [os.path.join(folder, f) for f in ["cluster_pixel_map.npy",
                                                   "adu_weighted_ccd_final.npy",
                                                   "ccd_redistributed.npy"]]
        if not all(os.path.exists(f) for f in files):
            print(f"Missing data for image {idx}, skipping.")
            continue

        # Load SPC and high-ADU data
        cluster_pixel_map, adu_weighted_ccd, ccd_redistributed = [np.load(f) for f in files]

        # Remove bright pixels identified during full-image summation
        remove_high_ADU_pixels(adu_weighted_ccd, ccd_redistributed, high_adu_positions)

        # Extract SPC and high-ADU hits
        spc_hits, high_adu_hits = extract_photon_hits(cluster_pixel_map, adu_weighted_ccd, ccd_redistributed)

        total_spc_photons_before_overlap += len(spc_hits)
        total_high_adu_photons += len(high_adu_hits)

        # Resolve spatial overlaps between SPC and high-ADU clusters
        clean_spc_hits, clean_high_adu_hits, num_removed_overlaps = resolve_overlaps(spc_hits, high_adu_hits)
        total_removed_overlaps += num_removed_overlaps
        total_spc_photons_after_overlap += len(clean_spc_hits)

        # Quantify photon counts per hit
        clean_spc_photons = assign_photon_counts(clean_spc_hits, high_adu=False)
        clean_high_adu_photons = assign_photon_counts(clean_high_adu_hits, high_adu=True)
        print(f"Assigned photon counts: SPC={len(clean_spc_photons)}, High-ADU={len(clean_high_adu_photons)}")

        # Combine all photon hits
        photon_hits_combined = np.vstack([clean_spc_photons, clean_high_adu_photons])
        photon_cluster_sizes = photon_hits_combined[:, 2].astype(int)

        # Save hit positions for CCD plot
        x_hits_all.extend(photon_hits_combined[:, 0])
        y_hits_all.extend(photon_hits_combined[:, 1])

        # Map (x, y) to energy and replicate by photon count
        photon_energies, _ = map_photon_energies(photon_hits_combined, optimized_params)
        photon_energy_weighted = np.repeat(photon_energies, photon_cluster_sizes)

        # Histogram raw photon counts
        hist_counts, _ = np.histogram(photon_energy_weighted, bins=ENERGY_BINS)

        # Apply solid angle correction
        corrected_intensity = hist_counts / Omega_E
        corrected_intensity[np.isnan(corrected_intensity)] = 0

        # Compute and propagate Poisson error
        raw_poisson_errors = np.sqrt(hist_counts)
        corrected_errors = raw_poisson_errors / Omega_E
        corrected_errors[np.isnan(corrected_errors)] = 0

        # Accumulate results
        accumulated_corrected_counts += corrected_intensity
        accumulated_errors_squared += corrected_errors ** 2

        # Debug: print a few overlap coordinates
        if num_removed_overlaps > 0:
            overlapping_coords = set(map(tuple, spc_hits[:, :2])) & set(map(tuple, high_adu_hits[:, :2]))
            print(f"[DEBUG] Found {num_removed_overlaps} overlaps. Example coordinates:")
            for overlap in list(overlapping_coords)[:5]:
                print(f"  Overlap at pixel: (np.float64({overlap[0]}), np.float64({overlap[1]}))")

        print(f"Image {idx} summary:")
        print(f"  SPC photon hits before overlap removal: {len(spc_hits)}")
        print(f"  High-ADU photon hits: {len(high_adu_hits)}")
        print(f"  Overlapping SPC photons removed: {num_removed_overlaps}")
        print(f"  Total photon hits after cleaning: {len(photon_hits_combined)}")

    # Final error propagation and bin center calculation
    total_corrected_errors = np.sqrt(accumulated_errors_squared)
    bin_centers = (ENERGY_BINS[:-1] + ENERGY_BINS[1:]) / 2

    # --- Final plots ---
    plot_ccd_hits(np.array(x_hits_all), np.array(y_hits_all))
    plot_solid_angle_vs_energy(ENERGY_BINS, Omega_E)
    plot_solid_angle_adjusted_spectrum(bin_centers, accumulated_corrected_counts, total_corrected_errors)

    # Compute scaled intensity and propagate normalization errors
    intensity = accumulated_corrected_counts * bin_centers
    intensity_errors = total_corrected_errors * bin_centers

    intensity_min, intensity_max = intensity.min(), intensity.max()
    intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
    intensity_errors_norm = intensity_errors / (intensity_max - intensity_min)

    plot_intensity_vs_energy(bin_centers, intensity_norm, intensity_errors_norm)
    plot_log_intensity_vs_energy(bin_centers, accumulated_corrected_counts)
    plot_normalized_spectrum(bin_centers, accumulated_corrected_counts * Omega_E, accumulated_corrected_counts)
    plot_extracted_spectrum_lineout(bin_centers, accumulated_corrected_counts * Omega_E)

    # --- Summary statistics ---
    print("\n=== Overall Summary Across All Images ===")
    print(f"Total SPC photons before overlap removal: {total_spc_photons_before_overlap}")
    print(f"Total SPC photons after overlap removal: {total_spc_photons_after_overlap}")
    print(f"Total High-ADU photons: {total_high_adu_photons}")
    print(f"Total overlaps removed: {total_removed_overlaps}")
    print(f"Total photon counts (ADU-weighted): {int(np.sum(accumulated_corrected_counts * Omega_E))}")

    def print_array_sample(name, arr, num=5):
        arr = np.asarray(arr)
        print(f"{name}: {arr[:num]} ... {arr[-num:]}" if arr.size > 2 * num else f"{name}: {arr}")

    print_array_sample("Corrected intensity (counts)", accumulated_corrected_counts)
    print_array_sample("Corrected Poisson errors (sqrt)", total_corrected_errors)

    percent_errors = 100 * total_corrected_errors / accumulated_corrected_counts
    percent_errors[np.isnan(percent_errors)] = 0
    percent_errors[np.isinf(percent_errors)] = 0
    print(f"Percent Poisson Error (%): min = {np.min(percent_errors):.3f}, max = {np.max(percent_errors):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spectral reconstruction with SPC and high-ADU processing.")
    parser.add_argument("file_name", nargs="?", default=HDF5_FILE, help="HDF5 file with CCD images.")
    parser.add_argument("--images", type=int, nargs='+', help="Indices of images to process (e.g. --images 3 5 8).")
    args = parser.parse_args()

    run_spectral_reconstruction(filename=args.file_name, image_indices=args.images)