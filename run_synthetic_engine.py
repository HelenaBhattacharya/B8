import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from synthetic_engine.detector import CCDDetector
from synthetic_engine.synthetic_generator import (
    place_cluster, synthetic_spectrum, build_energy_map
)

from spc_engine.spc import detect_photon_events
from spc_engine.high_ADU import apply_ADU_threshold, redistribute_charge
from bragg_engine.mapping import compute_energy_map
from spectral_reconstruction.spectral_processing import map_photon_energies


def main():
    # === Paths ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    detector_params_path = os.path.join(base_dir, "synthetic_engine", "detector_params.json")
    optimized_params_path = os.path.join(base_dir, "optimized_params.npy")
    output_dir = os.path.join(base_dir, "synthetic_dataset_example")
    os.makedirs(output_dir, exist_ok=True)

    # === Load detector + Bragg parameters ===
    detector = CCDDetector(detector_params_path)
    optimized_params = np.load(optimized_params_path)
    E_ij = build_energy_map(optimized_params)

    # === Generate spectrum intensity map ===
    intensity_map = synthetic_spectrum(E_ij)
    ny, nx = intensity_map.shape
    synthetic_ccd = np.zeros((ny, nx), dtype=np.float32)
    label_map = np.zeros((ny, nx), dtype=np.uint8)

    # === Place photon clusters based on spectrum ===
    placement_probs = intensity_map / np.sum(intensity_map)
    placement_cdf = np.cumsum(placement_probs.ravel())
    num_clusters = 2500
    composite_ratio = 0.1

    for _ in trange(num_clusters, desc="Placing photon clusters"):
        flat_idx = np.searchsorted(placement_cdf, np.random.rand())
        y0, x0 = np.unravel_index(flat_idx, (ny, nx))
        cluster_type = 'composite' if np.random.rand() < composite_ratio else 'single'

        place_cluster(
            synthetic_ccd,
            label_map,
            x0=x0,
            y0=y0,
            cluster_type=cluster_type
        )

    # === Convert ADU → electrons → simulate CCD ===
    photon_hits = [
        (y, x, synthetic_ccd[y, x] * detector.adc_gain)
        for y in range(ny)
        for x in range(nx)
        if synthetic_ccd[y, x] > 0
    ]
    bright_pixel_count = np.sum(synthetic_ccd > 100)
    print(f"Bright raw ADU pixels before detector model: {bright_pixel_count}")

    final_ccd = detector.acquire_frame(exposure_s=1.0, photon_hits=photon_hits)
    print("Max pixel ADU in CCD:", np.max(final_ccd))
    print("Mean ADU (raw):", np.mean(final_ccd))
    print("Number of pixels > 150 ADU:", np.sum(final_ccd > 150))

    # === Save outputs ===
    np.save(os.path.join(output_dir, "synthetic_ccd_image.npy"), final_ccd)
    np.save(os.path.join(output_dir, "ground_truth_label_map.npy"), label_map)
    np.save(os.path.join(output_dir, "photon_energy_map.npy"), E_ij)

    # === Plot: Synthetic CCD Image ===
    plt.figure(figsize=(8, 6))
    plt.imshow(final_ccd, cmap='inferno', origin='upper', vmin=60, vmax=np.percentile(final_ccd, 99.5))
    plt.title("Synthetic CCD Image (ADU + Pedestal + Noise)")
    plt.colorbar(label="ADU")
    plt.tight_layout()
    plt.show()

    # === Pedestal subtraction for SPC ===
    PEDESTAL_MU = 60
    ccd_sub = np.clip(final_ccd.astype(float) - PEDESTAL_MU, 0, None)

    # === Detect SPC photon events ===
    SIGMA_N = 9.5
    adu_map, cluster_map, photon_events, high_adu_clusters, _, thresholds = detect_photon_events(
        ccd_sub, sigma_N=SIGMA_N
    )
    print(f"\nSPC thresholds: Initial={thresholds[0]:.2f}, Secondary={thresholds[1]:.2f}, Single Photon ~{thresholds[2]}")

    # === Plot: Histogram of ADU Pixel Values (Raw CCD) ===
    plt.figure(figsize=(8, 5))
    plt.hist(final_ccd.ravel(), bins=300, range=(0, 500), color='darkblue', alpha=0.8)
    plt.axvline(PEDESTAL_MU + SIGMA_N * 1.5, color='red', linestyle='--', label='SPC T1 Threshold')
    plt.yscale('log')  # <-- log scale here
    plt.xlabel("Raw ADU Value")
    plt.ylabel("Pixel Count (log)")
    plt.title("Histogram of Pixel ADU Values (Raw CCD, Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot: CCD After Pedestal Subtraction + T1 Threshold ===
    ccd_t1 = ccd_sub.copy()
    T1 = thresholds[0]
    ccd_t1[ccd_t1 < T1] = 0

    plt.figure(figsize=(8, 6))
    plt.imshow(ccd_t1, cmap='inferno', origin='upper', vmin=0, vmax=np.percentile(ccd_t1, 99.5))
    plt.title("CCD After Pedestal Subtraction + T1 Threshold")
    plt.colorbar(label="ADU (Post-Subtraction)")
    plt.tight_layout()
    plt.show()

    # === High-ADU cluster extraction and redistribution ===
    high_adu_clusters = apply_ADU_threshold(high_adu_clusters, threshold_ADU=20)
    _, _, high_adu_map, _, _ = redistribute_charge(high_adu_clusters, ccd_sub.shape)

    # === Combine all photon hits ===
    photon_hits_detected = []
    counts_by_size = {}

    for size in [1, 2, 3, 4]:
        hits = photon_events[size]
        if hits:
            photon_hits_detected.extend([(y, x, adu) for y, x, adu in hits])
            counts_by_size[size] = len(hits)
        else:
            counts_by_size[size] = 0

    y_high, x_high = np.nonzero(high_adu_map)
    high_adu_hits = [(y, x, 1) for y, x in zip(y_high, x_high)]
    photon_hits_detected += high_adu_hits
    photon_hits_detected = np.array(photon_hits_detected)

    print(f"\nSPC hits: 1-px={counts_by_size[1]}, 2-px={counts_by_size[2]}, "
          f"3-px={counts_by_size[3]}, 4-px={counts_by_size[4]}")
    print(f"High-ADU hits: {len(high_adu_hits)}")
    print(f"Total photon hits detected: {len(photon_hits_detected)}")

    # === Energy mapping ===
    photon_energies, photon_adus = map_photon_energies(photon_hits_detected, optimized_params)

    # === Plot Spectrum ===
    ENERGY_BINS = np.arange(1100, 1601, 2)
    bin_centers = (ENERGY_BINS[:-1] + ENERGY_BINS[1:]) / 2
    photon_adus_clipped = np.clip(photon_adus, 1, 10)
    weighted_energies = np.repeat(photon_energies, photon_adus_clipped.astype(int))
    counts, _ = np.histogram(weighted_energies, bins=ENERGY_BINS)

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=2, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Photon Counts")
    plt.title("Synthetic Spectrum (SPC + High-ADU)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === Plot 2: Photon Hit Map ===
    plt.figure(figsize=(10, 10))
    for size, color in zip([1, 2, 3, 4], ['blue', 'green', 'orange', 'red']):
        hits = photon_events[size]
        if hits:
            arr = np.array([(y, x) for y, x, _ in hits])
            plt.scatter(arr[:, 1], arr[:, 0], s=0.5, color=color, label=f"SPC {size}-pixel")

    if len(high_adu_hits) > 0:
        arr = np.array([(y, x) for y, x, _ in high_adu_hits])
        plt.scatter(arr[:, 1], arr[:, 0], s=0.5, color="magenta", label="High-ADU")

    plt.gca().invert_yaxis()
    plt.xlabel("CCD X Position")
    plt.ylabel("CCD Y Position")
    plt.title("Photon Hit Map (SPC + High-ADU)")
    plt.legend(markerscale=5, loc='upper right')
    plt.tight_layout()
    plt.show()

    # === Load reference data from image 8 ===
    external_data_dir = os.path.join(base_dir, "8")
    cluster_map_file = os.path.join(external_data_dir, "cluster_pixel_map.npy")
    adu_weighted_file = os.path.join(external_data_dir, "adu_weighted_ccd_final.npy")
    redistributed_file = os.path.join(external_data_dir, "ccd_redistributed.npy")

    if all(os.path.exists(f) for f in [cluster_map_file, adu_weighted_file, redistributed_file]):
        cluster_pixel_map = np.load(cluster_map_file)
        adu_weighted_ccd = np.load(adu_weighted_file)
        ccd_redistributed = np.load(redistributed_file)

        # Remove high-ADU pixels (from preprocessing)
        from spectral_reconstruction.spectral_processing import remove_high_ADU_pixels, extract_photon_hits

        high_adu_positions = np.argwhere(adu_weighted_ccd > 1000)  # or whatever is used in your config
        remove_high_ADU_pixels(adu_weighted_ccd, ccd_redistributed, high_adu_positions)

        spc_hits, high_adu_hits = extract_photon_hits(cluster_pixel_map, adu_weighted_ccd, ccd_redistributed)
        photon_hits_combined = np.concatenate([spc_hits, high_adu_hits])
        photon_energies_ref, photon_adus_ref = map_photon_energies(photon_hits_combined, optimized_params)

        # Histogram for reference
        ENERGY_BINS = np.arange(1100, 1601, 2)
        bin_centers = (ENERGY_BINS[:-1] + ENERGY_BINS[1:]) / 2
        ref_weighted_energies = np.repeat(photon_energies_ref, photon_adus_ref.astype(int))
        ref_counts, _ = np.histogram(ref_weighted_energies, bins=ENERGY_BINS)

        # === Normalised Comparison Plot ===
        synthetic_norm = counts / np.max(counts)
        reference_norm = ref_counts / np.max(ref_counts)

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, synthetic_norm, label="Synthetic Reconstructed", lw=1, color='purple')
        plt.plot(bin_centers, reference_norm, label="Image 8 Reference", lw=1, linestyle='--', color='blue')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Normalised Intensity")
        plt.title("Synthetic vs Image 8 Spectrum")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("Missing one or more reference files for image 8.")

    print("\nSynthetic CCD generation and analysis complete.")


if __name__ == "__main__":
    main()


