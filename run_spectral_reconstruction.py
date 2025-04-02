import argparse
import numpy as np
import os
from bragg_engine.preprocessing import sum_all_ccd_images
from bragg_engine.mapping import compute_energy_map
from spc_engine.intensity_error import poisson_error
from spectral_reconstruction.spectral_processing import (
    remove_high_ADU_pixels,
    extract_photon_hits,
    resolve_overlaps,
    map_photon_energies,
    compute_solid_angle_map,
    solid_angle_correction,
    simple_solid_angle_per_bin)

from spectral_reconstruction.visualisation import (
    plot_photon_counts_vs_energy,
    plot_solid_angle_adjusted_spectrum,
    plot_intensity_vs_energy,
    plot_log_intensity_vs_energy,
    plot_normalized_spectrum,
    plot_ccd_hits, plot_extracted_spectrum_lineout,
    plot_solid_angle_vs_energy
)
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
    SPC_ADU_SINGLE_PHOTON)

ENERGY_BINS = np.arange(ENERGY_MIN, ENERGY_MAX + BIN_WIDTH, BIN_WIDTH)


def run_spectral_reconstruction(filename=HDF5_FILE, image_indices=None):
    print(f"Running spectral reconstruction for file: {filename}")

    optimized_params = np.load(OPTIMIZED_PARAMS_FILE, allow_pickle=True)
    summed_image, num_high_adu, high_adu_positions = sum_all_ccd_images(filename)
    print(f"High-ADU pixels identified: {num_high_adu}")

    images_to_process = image_indices if image_indices is not None else range(20)

    total_removed_overlaps = 0
    total_spc_photons_before_overlap = 0
    total_spc_photons_after_overlap = 0
    total_high_adu_photons = 0
    x_hits_all, y_hits_all = [], []

    accumulated_corrected_counts = np.zeros(len(ENERGY_BINS) - 1)
    accumulated_errors_squared = np.zeros(len(ENERGY_BINS) - 1)

    E_ij, _, _ = compute_energy_map(optimized_params)
    Omega_ij = compute_solid_angle_map(optimized_params)
    Omega_E = simple_solid_angle_per_bin(E_ij, Omega_ij, ENERGY_BINS)

    # --- Explicit ADU-to-Photon assignment function ---
    def assign_photon_counts(hits_array, high_adu=False):
        photon_counts = []
        T_secondary = SPC_ADU_SINGLE_PHOTON - SPC_SIGMA_N  # dynamically calculated clearly

        for hit in hits_array:
            x, y, adu = hit
            if not high_adu:
                # SPC events explicitly handled here (single-photon)
                if T_secondary <= adu < HIGH_ADU_CLUSTER_THRESHOLDS[1]:
                    photons = 1
                else:
                    photons = 0  # exclude noise explicitly
            else:
                # High ADU events explicitly (multi-photon)
                if adu < HIGH_ADU_CLUSTER_THRESHOLDS[2]:
                    photons = 2
                elif adu < HIGH_ADU_CLUSTER_THRESHOLDS[3]:
                    photons = 3
                else:
                    photons = 4
            if photons > 0:
                photon_counts.append([x, y, photons])
        return np.array(photon_counts)

    # --- end of explicit assignment function ---

    for idx in images_to_process:
        print(f"\nProcessing image {idx}...")

        folder = str(idx)
        files = [os.path.join(folder, f) for f in ["cluster_pixel_map.npy",
                                                   "adu_weighted_ccd_final.npy",
                                                   "ccd_redistributed.npy"]]
        if not all(os.path.exists(f) for f in files):
            print(f"Missing data for image {idx}, skipping.")
            continue

        cluster_pixel_map, adu_weighted_ccd, ccd_redistributed = [np.load(f) for f in files]

        initial_nonzero_adu = np.count_nonzero(adu_weighted_ccd)
        initial_nonzero_redistributed = np.count_nonzero(ccd_redistributed)

        remove_high_ADU_pixels(adu_weighted_ccd, ccd_redistributed, high_adu_positions)

        final_nonzero_adu = np.count_nonzero(adu_weighted_ccd)
        final_nonzero_redistributed = np.count_nonzero(ccd_redistributed)

        print(f"Removed high-ADU pixels:")
        print(f"  ADU-weighted CCD: {initial_nonzero_adu - final_nonzero_adu} pixels set to 0.")
        print(f"  Redistributed CCD: {initial_nonzero_redistributed - final_nonzero_redistributed} pixels set to 0.")

        spc_hits, high_adu_hits = extract_photon_hits(cluster_pixel_map, adu_weighted_ccd, ccd_redistributed)

        total_spc_photons_before_overlap += len(spc_hits)
        total_high_adu_photons += len(high_adu_hits)

        clean_spc_hits, clean_high_adu_hits, num_removed_overlaps = resolve_overlaps(spc_hits, high_adu_hits)

        total_removed_overlaps += num_removed_overlaps
        total_spc_photons_after_overlap += len(clean_spc_hits)

        # Explicit photon assignment now clearly done here:
        clean_spc_photons = assign_photon_counts(clean_spc_hits, high_adu=False)
        clean_high_adu_photons = assign_photon_counts(clean_high_adu_hits, high_adu=True)
        print(f"Assigned photon counts: SPC={len(clean_spc_photons)}, High-ADU={len(clean_high_adu_photons)}")

        # Combine explicitly assigned photon hits:
        photon_hits_combined = np.vstack([clean_spc_photons, clean_high_adu_photons])

        # Explicit extraction of positions:
        x_hits_all.extend(photon_hits_combined[:, 0])
        y_hits_all.extend(photon_hits_combined[:, 1])

        # Explicitly use assigned photon counts now:
        photon_cluster_sizes = photon_hits_combined[:, 2].astype(int)

        photon_energies, _ = map_photon_energies(photon_hits_combined, optimized_params)
        photon_energy_weighted = np.repeat(photon_energies, photon_cluster_sizes)

        # Histogram counts per image
        hist_counts, _ = np.histogram(photon_energy_weighted, bins=ENERGY_BINS)

        # Solid angle correction per image
        corrected_intensity = hist_counts / Omega_E
        corrected_intensity[np.isnan(corrected_intensity)] = 0

        # Poisson errors per image
        raw_poisson_errors = np.sqrt(hist_counts)
        corrected_errors = raw_poisson_errors / Omega_E
        corrected_errors[np.isnan(corrected_errors)] = 0

        accumulated_corrected_counts += corrected_intensity
        accumulated_errors_squared += corrected_errors ** 2

        # Verbose overlap details (matched exactly)
        if num_removed_overlaps > 0:
            overlapping_coords = set(map(tuple, spc_hits[:, :2])) & set(map(tuple, high_adu_hits[:, :2]))
            print(f"[DEBUG] Found {num_removed_overlaps} overlaps. Example coordinates:")
            for overlap in list(overlapping_coords)[:5]:
                print(f"  Overlap at pixel: (np.float64({overlap[0]}), np.float64({overlap[1]}))")

        print(f"Image {idx} summary:")
        print(f"  SPC photon hits before overlap removal: {len(spc_hits)}")
        print(f"  High-ADU photon hits: {len(high_adu_hits)}")
        print(f"  Overlapping SPC photons removed: {num_removed_overlaps}")
        print(f"  SPC photon hits after overlap removal: {len(clean_spc_hits)}")
        print(f"  Total photons (SPC + high-ADU) after cleaning: {len(photon_hits_combined)}")

    total_corrected_errors = np.sqrt(accumulated_errors_squared)

    bin_centers = (ENERGY_BINS[:-1] + ENERGY_BINS[1:]) / 2
    plot_ccd_hits(np.array(x_hits_all), np.array(y_hits_all))
    plot_solid_angle_vs_energy(ENERGY_BINS, Omega_E)
    plot_solid_angle_adjusted_spectrum(bin_centers, accumulated_corrected_counts, total_corrected_errors)
    plot_intensity_vs_energy(bin_centers, accumulated_corrected_counts * bin_centers, total_corrected_errors * bin_centers)
    plot_log_intensity_vs_energy(bin_centers, accumulated_corrected_counts)
    plot_normalized_spectrum(bin_centers, accumulated_corrected_counts * Omega_E, accumulated_corrected_counts)
    plot_extracted_spectrum_lineout(bin_centers, accumulated_corrected_counts * Omega_E)

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


#
# import argparse
# import numpy as np
# import os
# from bragg_engine.preprocessing import sum_all_ccd_images
# from bragg_engine.mapping import compute_energy_map
# from spc_engine.intensity_error import poisson_error
# from spectral_reconstruction.spectral_processing import (
#     remove_high_ADU_pixels,
#     extract_photon_hits,
#     resolve_overlaps,
#     map_photon_energies,
#     compute_solid_angle_map,
#     solid_angle_correction,
#     fractional_solid_angle_per_bin
# )
#
# from spectral_reconstruction.visualisation import (
#     plot_photon_counts_vs_energy,
#     plot_solid_angle_adjusted_spectrum,
#     plot_intensity_vs_energy,
#     plot_log_intensity_vs_energy,
#     plot_normalized_spectrum,
#     plot_ccd_hits, plot_extracted_spectrum_lineout,
#     plot_solid_angle_vs_energy
# )
# from config import (
#     HDF5_FILE,
#     CCD_SHAPE,
#     OPTIMIZED_PARAMS_FILE,
#     ENERGY_MIN,
#     ENERGY_MAX,
#     BIN_WIDTH,
#     PIXEL_SIZE)
#
#
#
# ENERGY_BINS = np.arange(ENERGY_MIN, ENERGY_MAX + BIN_WIDTH, BIN_WIDTH)
#
#
# def run_spectral_reconstruction(filename=HDF5_FILE, image_index=None):
#     print(f"Running spectral reconstruction for file: {filename}")
#
#     optimized_params = np.load(OPTIMIZED_PARAMS_FILE, allow_pickle=True)
#
#     summed_image, num_high_adu, high_adu_positions = sum_all_ccd_images(filename)
#     print(f"High-ADU pixels identified: {num_high_adu}")
#
#     images_to_process = [image_index] if image_index is not None else range(20)
#     photon_energy_weighted_all = []
#     total_removed_overlaps = 0
#     total_spc_photons_before_overlap = 0
#     total_spc_photons_after_overlap = 0
#     total_high_adu_photons = 0
#     x_hits_all, y_hits_all = [], []
#
#     for idx in images_to_process:
#         print(f"\nProcessing image {idx}...")
#
#         folder = str(idx)
#         cluster_map_file = os.path.join(folder, "cluster_pixel_map.npy")
#         adu_weighted_file = os.path.join(folder, "adu_weighted_ccd_final.npy")
#         redistributed_file = os.path.join(folder, "ccd_redistributed.npy")
#
#         if not all(os.path.exists(f) for f in [cluster_map_file, adu_weighted_file, redistributed_file]):
#             print(f"Missing data for image {idx}, skipping.")
#             continue
#
#         cluster_pixel_map = np.load(cluster_map_file)
#         adu_weighted_ccd = np.load(adu_weighted_file)
#         ccd_redistributed = np.load(redistributed_file)
#
#         remove_high_ADU_pixels(adu_weighted_ccd, ccd_redistributed, high_adu_positions)
#
#         spc_hits, high_adu_hits = extract_photon_hits(cluster_pixel_map, adu_weighted_ccd, ccd_redistributed)
#
#         total_spc_photons_before_overlap += len(spc_hits)
#         total_high_adu_photons += len(high_adu_hits)
#
#         clean_spc_hits, clean_high_adu_hits, num_removed_overlaps = resolve_overlaps(spc_hits, high_adu_hits)
#
#         total_removed_overlaps += num_removed_overlaps
#         total_spc_photons_after_overlap += len(clean_spc_hits)
#
#         photon_hits_combined = np.concatenate([clean_spc_hits, clean_high_adu_hits])
#
#         # Accumulate hits across images correctly:
#         x_hits_all.extend(photon_hits_combined[:, 1])
#         y_hits_all.extend(photon_hits_combined[:, 0])
#
#         photon_energies, photon_adus = map_photon_energies(photon_hits_combined, optimized_params)
#         photon_energy_weighted_all.append(np.repeat(photon_energies, photon_adus.astype(int)))
#
#         # Individual image print statements:
#         print(f"Image {idx} summary:")
#         print(f"  SPC photon hits before overlap removal: {len(spc_hits)}")
#         print(f"  High-ADU photon hits: {len(high_adu_hits)}")
#         print(f"  Overlapping SPC photons removed: {num_removed_overlaps}")
#         print(f"  SPC photon hits after overlap removal: {len(clean_spc_hits)}")
#         print(f"  Total photons (SPC + high-ADU) after cleaning: {len(photon_hits_combined)}")
#
#     if photon_energy_weighted_all:
#         photon_energy_weighted_total = np.concatenate(photon_energy_weighted_all)
#         hist_counts_total, _ = np.histogram(photon_energy_weighted_total, bins=ENERGY_BINS)
#
#         # Convert lists to numpy arrays for plotting
#         x_hits_all = np.array(x_hits_all)
#         y_hits_all = np.array(y_hits_all)
#
#         plot_ccd_hits(x_hits_all, y_hits_all)
#
#         print("\n=== Overall Summary Across All Images ===")
#         print(f"Total SPC photons before overlap removal: {total_spc_photons_before_overlap}")
#         print(f"Total SPC photons after overlap removal: {total_spc_photons_after_overlap}")
#         print(f"Total High-ADU photons: {total_high_adu_photons}")
#         print(f"Total overlaps removed: {total_removed_overlaps}")
#         print(f"Total photon counts (ADU-weighted): {hist_counts_total.sum()}")
#
#         # Solid angle correction:
#         E_ij, x_prime, y_prime = compute_energy_map(optimized_params)
#         Omega_ij = compute_solid_angle_map(optimized_params)
#
#         corrected_intensity = solid_angle_correction(hist_counts_total, ENERGY_BINS, E_ij, Omega_ij)
#
#         # === Poisson Error Calculation ===
#         # Correct and explicit Poisson error calculation:
#
#         raw_counts = hist_counts_total
#         raw_poisson_errors = np.sqrt(raw_counts)
#
#         bin_width = ENERGY_BINS[1] - ENERGY_BINS[0]
#
#         # explicitly for visualization purposes only
#         Omega_E = fractional_solid_angle_per_bin(E_ij, Omega_ij, ENERGY_BINS)
#         plot_solid_angle_vs_energy(ENERGY_BINS, Omega_E)
#
#         # Photon Count Error (solid-angle adjusted):
#         corrected_count_errors = raw_poisson_errors / Omega_E
#         corrected_count_errors[np.isnan(corrected_count_errors)] = 0
#
#         # Intensity Error (scaled by bin energy):
#         bin_centers = (ENERGY_BINS[:-1] + ENERGY_BINS[1:]) / 2
#         corrected_intensity_errors = corrected_count_errors * bin_centers
#
#         # Compute percentage Poisson error (avoid divide-by-zero)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             percent_errors = 100 * corrected_intensity_errors / (corrected_intensity * bin_centers)
#             percent_errors[np.isnan(percent_errors)] = 0
#             percent_errors[np.isinf(percent_errors)] = 0
#
#         # === Visualization Calls ===
#         plot_photon_counts_vs_energy(bin_centers, hist_counts_total, BIN_WIDTH,
#                                      title="Extracted X-ray Spectrum (Before Solid Angle Correction)")
#
#         plot_solid_angle_adjusted_spectrum(
#             bin_centers,
#             corrected_intensity,
#             corrected_errors=corrected_count_errors  # sqrt(N)/Omega
#         )
#         plot_intensity_vs_energy(
#             bin_centers,
#             corrected_intensity * bin_centers,  # intensity = count × energy
#             intensity_errors=corrected_intensity_errors  # sqrt(N)*E/Omega
#         )
#         plot_log_intensity_vs_energy(bin_centers, corrected_intensity)
#         plot_normalized_spectrum(bin_centers, hist_counts_total, corrected_intensity)
#         plot_extracted_spectrum_lineout(bin_centers,hist_counts_total)
#
#         def print_array_sample(name, arr, num=5):
#             """Prints the first and last few elements of an array."""
#             arr = np.asarray(arr)
#             if arr.size <= 2 * num:
#                 print(f"{name}: {arr}")
#             else:
#                 print(f"{name}: {arr[:num]} ... {arr[-num:]}")
#
#         # Usage:
#         print_array_sample("Photon counts per bin (N)", raw_counts)
#         print_array_sample("Poisson errors (sqrt(N))", raw_poisson_errors)
#         print_array_sample("Summed solid-angle per bin (Omega_E)", Omega_E)
#         print_array_sample("Corrected errors (sqrt(N)/Omega)", corrected_count_errors)
#         print_array_sample("Corrected intensity errors (sqrt(N)*E/Omega)", corrected_intensity_errors)
#         print(f"Percent Poisson Error (%): min = {np.min(percent_errors):.3f}, max = {np.max(percent_errors):.3f}")
#
#     else:
#         print("No photon data collected. Check inputs.")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run spectral reconstruction with SPC and high-ADU processing.")
#     parser.add_argument("file_name", nargs="?", default=HDF5_FILE, help="HDF5 file with CCD images.")
#     parser.add_argument("--image_index", type=int, help="Index of single image to process (optional).")
#
#     args = parser.parse_args()
#     run_spectral_reconstruction(filename=args.file_name, image_index=args.image_index)