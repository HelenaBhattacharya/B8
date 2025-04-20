"""
run_spc_engine.py

Runs the Single-Photon Counting (SPC) engine for CCD image processing.

This script performs:
- Gaussian pedestal subtraction
- Single-photon cluster detection (SPC)
- High-ADU cluster thresholding and charge redistribution
- Optional batch or single-image processing
- Visualizations: CCD images, histograms, and detected photon hits

Modules:
- `pedestal`: pedestal subtraction using Gaussian fitting
- `spc`: SPC event detection and cluster extraction
- `high_ADU`: charge redistribution for bright clusters
- `visualisation`: histogram and hit map generation

Usage:
    python run_spc_engine.py [file_name] [--single_image N]
"""

import argparse
import numpy as np
import os
# Module imports for data processing and visualization
from spc_engine.pedestal import process_pedestal_correction, load_ccd_images
from spc_engine.visualisation import (
    plot_full_histogram,
    plot_corrected_histogram,
    plot_original_ccd_image,
    plot_corrected_ccd_image,
    plot_photon_hits,
    plot_high_ADU_clusters
)
from spc_engine.spc import detect_photon_events, save_spc_results
from spc_engine.high_ADU import (
    load_high_ADU_clusters,
    apply_ADU_threshold,
    redistribute_charge,
    save_photon_hit_data,
    save_ccd_image
)

from config import HDF5_FILE, CCD_SHAPE, HIGH_ADU_PIXEL_THRESHOLD, MAX_IMAGES

# Main control function: process a single image or all images
def run_spc_engine(file_name=HDF5_FILE, single_image=None):
    """
    Run the SPC engine on a single CCD image or a full dataset.

    Parameters:
        file_name (str): Path to the CCD HDF5 file.
        single_image (int or None): If provided, process only that image index.
                                    If None, process all images up to MAX_IMAGES.
    """
    print(f"Using file: {file_name}")
    plot_queue = []

    if single_image is not None:
        print(f"Processing single image index: {single_image}")
        process_single_image(file_name, single_image, plot_queue)
    else:
        print("Processing all images...")
        process_all_images(file_name, plot_queue)

    print("\nAll image processing complete. Now generating visualizations...")
    plot_all_results(plot_queue)

# Process one CCD image: apply pedestal correction, detect SPC events, handle high ADU clusters
def process_single_image(file_name, image_index, plot_queue):
    """
    Process a single CCD image:
    - Apply pedestal subtraction
    - Detect single-photon events (SPC)
    - Handle high-ADU photon clusters
    - Store processed data for visualization

    Parameters:
        file_name (str): Path to HDF5 file.
        image_index (int): Index of the image to process.
        plot_queue (list): Collector for visualization data.
    """
    print(f"\nProcessing single image (index {image_index})...\n")

    images = load_ccd_images(file_name, num_images=image_index + 1)
    image = images[image_index]
    pixel_values = image.flatten()

    # Subtract pedestal using Gaussian fit
    corrected_image, mu_fit, sigma_fit = process_pedestal_correction(file_name, image_index=image_index)
    corrected_pixel_values = corrected_image.flatten()

    print(f"Fitted Pedestal Mean (mu): {mu_fit:.2f} ADU")
    print(f"Fitted Pedestal Std (sigma): {sigma_fit:.2f} ADU")
    print("Pedestal subtracted from CCD data.")

    # Detect single-photon clusters and extract thresholds
    adu_map, cluster_pixel_map, photon_events, high_ADU_clusters, filtered_high_ADU_clusters, thresholds = detect_photon_events(
        corrected_image, sigma_N=sigma_fit
    )
    T_initial, T_secondary, _ = thresholds
    print(f"T_initial (T1): {T_initial:.2f} ADU, T_secondary (T2): {T_secondary:.2f} ADU")

    # Save cluster and ADU maps
    save_spc_results(adu_map, cluster_pixel_map, high_ADU_clusters, photon_events, image_index=image_index)

    print(f"SPC completed for image {image_index}.")

    # Post-process high-ADU clusters and queue data for plotting
    high_ADU_processing(
        image_index, pixel_values, mu_fit, sigma_fit,
        image, corrected_image, corrected_pixel_values,
        photon_events, plot_queue
    )

# Process all CCD images: same as above, but iterates over all frames
def process_all_images(file_name, plot_queue):
    """
    Process all CCD images sequentially (up to MAX_IMAGES).

    Parameters:
        file_name (str): Path to HDF5 file.
        plot_queue (list): Collector for visualization data.
    """

    print("\nProcessing all 20 images sequentially...\n")

    images = load_ccd_images(file_name, num_images=MAX_IMAGES)

    for i, image in enumerate(images):
        print(f"\nProcessing image {i}...\n")

        pixel_values = image.flatten()

        corrected_image, mu_fit, sigma_fit = process_pedestal_correction(file_name, image_index=i)
        corrected_pixel_values = corrected_image.flatten()

        print(f"Image {i}: Fitted Pedestal Mean = {mu_fit:.2f}, Std = {sigma_fit:.2f}")

        adu_map, cluster_pixel_map, photon_events, high_ADU_clusters, filtered_high_ADU_clusters, thresholds = detect_photon_events(
            corrected_image, sigma_N=sigma_fit
        )
        T_initial, T_secondary, _ = thresholds
        print(f"Image {i} thresholds: T_initial (T1) = {T_initial:.2f} ADU, T_secondary (T2) = {T_secondary:.2f} ADU")

        save_spc_results(adu_map, cluster_pixel_map, high_ADU_clusters, photon_events, image_index=i)

        print(f"SPC completed for image {i}.")

        high_ADU_processing(
            i, pixel_values, mu_fit, sigma_fit,
            image, corrected_image, corrected_pixel_values,
            photon_events, plot_queue
        )

    print("\nSPC and high ADU processing completed for all images.")

# Handle high-ADU clusters: apply threshold, redistribute charge, compute photon counts
def high_ADU_processing(image_index, pixel_values, mu_fit, sigma_fit, original_image, corrected_image, corrected_pixel_values, photon_events, plot_queue):
    """
        Processes high-ADU photon clusters for a single CCD image.

        - Loads high-ADU clusters from disk
        - Applies ADU thresholding
        - Redistributes charge to estimate photon counts
        - Saves outputs to file
        - Updates plot queue with SPC and high-ADU hit data

        Parameters:
            image_index (int): Index of the processed image.
            pixel_values (np.ndarray): Raw pixel values before correction.
            mu_fit (float): Mean of the pedestal fit.
            sigma_fit (float): Std deviation of the pedestal fit.
            original_image (np.ndarray): Raw CCD image.
            corrected_image (np.ndarray): CCD image after pedestal subtraction.
            corrected_pixel_values (np.ndarray): Flattened corrected pixel values.
            photon_events (dict): SPC photon hits categorized by cluster size.
            plot_queue (list): Shared list to append data for visualization.
        """
    folder = str(image_index)
    high_adu_clusters = load_high_ADU_clusters(os.path.join(folder, "high_ADU_clusters.pkl"))
    high_adu_clusters = apply_ADU_threshold(high_adu_clusters, threshold_ADU=HIGH_ADU_PIXEL_THRESHOLD)

    # Convert high-ADU clusters to estimated photon hits
    ccd_thresholded, ccd_redistributed, ccd_photon_hits, num_photon_hits, photon_counts = redistribute_charge(
        high_adu_clusters, ccd_size=CCD_SHAPE
    )

    # Save redistributed hits and CCD data
    save_photon_hit_data(high_adu_clusters, os.path.join(folder, "photon_hits_charge_redistributed.csv"))
    save_ccd_image(ccd_redistributed, os.path.join(folder, "ccd_redistributed.npy"))

    print("\nSPC Single-Photon Clusters:")
    for size in range(1, 5):
        spc_count = len(photon_events[size])
        print(f"  {size}-pixel SPC clusters: {spc_count}")

    print("\nHigh-ADU Photon Clusters (by assigned photon counts):")
    for size in range(1, 5):
        adu_count = photon_counts[size]
        print(f"  {size}-photon high-ADU clusters: {adu_count}")

    total_spc_hits = sum(len(photon_events[size]) for size in range(1, 5))
    total_high_adu_hits = sum(size * photon_counts[size] for size in range(1, 5))

    print(f"\nTotal SPC Photon Hits: {total_spc_hits}")
    print(f"Total High-ADU Photon Hits: {total_high_adu_hits}")
    print(f"Combined Total Photon Hits: {total_spc_hits + total_high_adu_hits}")

    # Store relevant data for visualization
    plot_queue.append({
        "image_index": image_index,
        "pixel_values": pixel_values,
        "corrected_pixel_values": corrected_pixel_values,
        "original_image": original_image,
        "corrected_image": corrected_image,
        "photon_events": photon_events,
        "ccd_photon_hits": ccd_photon_hits,
        "mu_fit": mu_fit,
        "sigma_fit": sigma_fit
    })

# Generate final plots after all image processing
def plot_all_results(plot_queue):
    """
    Generates diagnostic plots for all processed images.

    For each image in the queue, this function produces:
    - Full pixel histogram with pedestal fit
    - Histogram after pedestal subtraction
    - Raw and corrected CCD images
    - Photon hit maps (SPC and high-ADU clusters)

    Parameters:
        plot_queue (list): List of dictionaries containing image data,
                           photon events, and fit results.
    """
    for entry in plot_queue:
        i = entry["image_index"]
        bins = np.arange(0, np.max(entry["pixel_values"]), 2)

        plot_full_histogram(entry["pixel_values"], entry["mu_fit"], entry["sigma_fit"], bins)
        plot_corrected_histogram(entry["corrected_pixel_values"], bins)
        plot_original_ccd_image(entry["original_image"])
        plot_corrected_ccd_image(entry["corrected_image"])
        plot_photon_hits(entry["photon_events"], image_index=i)
        plot_high_ADU_clusters(entry["ccd_photon_hits"], image_index=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPC engine with pedestal subtraction.")
    parser.add_argument("file_name", type=str, nargs="?", default=HDF5_FILE, help="Path to CCD HDF5 file.")
    parser.add_argument("--single_image", type=int, help="Process a single image index.")
    args = parser.parse_args()

    run_spc_engine(file_name=args.file_name, single_image=args.single_image)
