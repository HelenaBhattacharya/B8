import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from config import HIGH_ADU_PIXEL_THRESHOLD, HIGH_ADU_CLUSTER_THRESHOLDS

def load_high_ADU_clusters(file_path):
    """Load high ADU clusters from a Pickle file."""
    with open(file_path, "rb") as f:
        high_adu_clusters = pickle.load(f)
    return high_adu_clusters

def apply_ADU_threshold(high_adu_clusters, threshold_ADU=HIGH_ADU_PIXEL_THRESHOLD):
    """Apply 20 ADU threshold to clusters and recalculate total ADU."""
    for cluster in high_adu_clusters:
        cluster["pixels"] = [(x, y, adu) for x, y, adu in cluster["pixels"] if adu >= threshold_ADU]
        cluster["total_ADU"] = sum(adu for _, _, adu in cluster["pixels"])
    return high_adu_clusters

def redistribute_charge(high_adu_clusters, ccd_size=(2048, 2048)):
    """Redistribute charge based on total ADU and assign photons."""
    ccd_thresholded = np.zeros(ccd_size, dtype=np.float32)
    ccd_redistributed = np.zeros(ccd_size, dtype=np.float32)
    ccd_photon_hits = np.zeros(ccd_size, dtype=np.int32)

    num_photon_hits = 0
    photon_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for cluster in high_adu_clusters:
        pixel_list = cluster["pixels"]
        if not pixel_list:
            continue

        total_adu = cluster["total_ADU"]

        if total_adu < HIGH_ADU_CLUSTER_THRESHOLDS[1]:
            num_photons = 1
        elif total_adu < HIGH_ADU_CLUSTER_THRESHOLDS[2]:
            num_photons = 2
        elif total_adu < HIGH_ADU_CLUSTER_THRESHOLDS[3]:
            num_photons = 3
        else:
            num_photons = 4

        photon_counts[num_photons] += 1
        num_photon_hits += num_photons

        sorted_pixels = sorted(pixel_list, key=lambda p: p[2], reverse=True)

        for i in range(min(num_photons, len(sorted_pixels))):
            x, y, _ = sorted_pixels[i]
            assigned_adu = total_adu / num_photons
            ccd_redistributed[x, y] = assigned_adu
            ccd_photon_hits[x, y] = 1

        for x, y, adu in pixel_list:
            ccd_thresholded[x, y] = adu

    return ccd_thresholded, ccd_redistributed, ccd_photon_hits, num_photon_hits, photon_counts

def save_photon_hit_data(high_adu_clusters, csv_filename):
    """Save photon hit data to CSV."""
    photon_hit_data = []
    for cluster in high_adu_clusters:
        pixel_list = cluster["pixels"]
        total_adu = cluster["total_ADU"]
        num_photons = int(round(total_adu / 100))

        for x, y, adu in pixel_list:
            photon_hit_data.append([x, y, num_photons, adu])

    df = pd.DataFrame(photon_hit_data, columns=["X", "Y", "Photon Count", "ADU Value"])
    df.to_csv(csv_filename, index=False)
    print(f"Photon hit data saved to '{csv_filename}' ({df.shape[0]} entries).")

def save_ccd_image(ccd_redistributed, npy_filename):
    """Save charge-redistributed CCD to NPY file."""
    np.save(npy_filename, ccd_redistributed)
    print(f"Charge-redistributed CCD saved to '{npy_filename}'.")


def process_high_ADU(image_index, ccd_size=(2048,2048), plot_results=True):
    folder = str(image_index)
    os.makedirs(folder, exist_ok=True)

    high_adu_file = os.path.join(folder, "high_ADU_clusters.pkl")
    csv_filename = os.path.join(folder, "photon_hits_charge_redistributed.csv")
    npy_filename = os.path.join(folder, "ccd_redistributed.npy")

    # Load and process clusters
    high_adu_clusters = load_high_ADU_clusters(high_adu_file)
    high_adu_clusters = apply_ADU_threshold(high_adu_clusters, threshold_ADU=20)
    ccd_thresholded, ccd_redistributed, ccd_photon_hits, num_photon_hits, photon_counts = redistribute_charge(high_adu_clusters, ccd_size=ccd_size)

    # Save results
    save_photon_hit_data(high_adu_clusters, csv_filename)
    save_ccd_image(ccd_redistributed, npy_filename)

    return num_photon_hits, photon_counts