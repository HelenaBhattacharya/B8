import argparse
import numpy as np
import os
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

CCD_SIZE = CCD_SHAPE

def run_spc_engine(file_name=HDF5_FILE, single_image=None):
    print(f"Using file: {file_name}")
    if single_image is not None:
        print(f"Processing single image index: {single_image}")
        process_single_image(file_name, single_image)
    else:
        print(" Processing all images...")
        process_all_images(file_name)

def process_single_image(file_name, image_index):
    print(f"\nProcessing single image (index {image_index})...\n")

    images = load_ccd_images(file_name, num_images=image_index + 1)
    image = images[image_index]
    pixel_values = image.flatten()

    mu_fit, sigma_fit = process_pedestal_correction(file_name, image_index=image_index)
    corrected_image = image - mu_fit
    corrected_pixel_values = corrected_image.flatten()

    print(f"Fitted Pedestal Mean (mu): {mu_fit:.2f} ADU")
    print(f"Fitted Pedestal Std (sigma): {sigma_fit:.2f} ADU")
    print("Pedestal subtracted from CCD data.")

    # Run SPC algorithm
    adu_map, cluster_pixel_map, photon_events, high_ADU_clusters, filtered_high_ADU_clusters, thresholds = detect_photon_events(corrected_image)
    save_spc_results(adu_map, cluster_pixel_map, high_ADU_clusters, photon_events, image_index=image_index)

    print(f"SPC completed for image {image_index}.")

    # High-ADU processing
    high_ADU_processing(image_index, pixel_values, mu_fit, sigma_fit, image, corrected_image, corrected_pixel_values, photon_events)

def process_all_images(file_name):
    print("\nProcessing all 20 images sequentially...\n")

    images = load_ccd_images(file_name, num_images=MAX_IMAGES)

    for i, image in enumerate(images):
        print(f"\nProcessing image {i}...\n")

        pixel_values = image.flatten()

        mu_fit, sigma_fit = process_pedestal_correction(file_name, image_index=i)
        corrected_image = image - mu_fit
        corrected_pixel_values = corrected_image.flatten()

        print(f"Image {i}: Fitted Pedestal Mean = {mu_fit:.2f}, Std = {sigma_fit:.2f}")

        # Run SPC algorithm
        adu_map, cluster_pixel_map, photon_events, high_ADU_clusters, filtered_high_ADU_clusters, thresholds = detect_photon_events(corrected_image)
        save_spc_results(adu_map, cluster_pixel_map, high_ADU_clusters, photon_events, image_index=i)

        print(f"SPC completed for image {i}.")

        # High-ADU processing
        high_ADU_processing(i, pixel_values, mu_fit, sigma_fit, image, corrected_image, corrected_pixel_values, photon_events)

    print("\nSPC and high ADU processing completed for all images.")

def high_ADU_processing(image_index, pixel_values, mu_fit, sigma_fit, original_image, corrected_image, corrected_pixel_values, photon_events):
    folder = str(image_index)

    # Load and filter high-ADU clusters
    high_adu_clusters = load_high_ADU_clusters(os.path.join(folder, "high_ADU_clusters.pkl"))
    high_adu_clusters = apply_ADU_threshold(high_adu_clusters, threshold_ADU=HIGH_ADU_PIXEL_THRESHOLD)

    # Charge redistribution
    ccd_thresholded, ccd_redistributed, ccd_photon_hits, num_photon_hits, photon_counts = redistribute_charge(high_adu_clusters, ccd_size=CCD_SIZE)

    # Save results
    save_photon_hit_data(high_adu_clusters, os.path.join(folder, "photon_hits_charge_redistributed.csv"))
    save_ccd_image(ccd_redistributed, os.path.join(folder, "ccd_redistributed.npy"))

    # SPC Photon Hit Clusters
    print("\nSPC Single-Photon Clusters:")
    for size in range(1, 5):
        spc_count = len(photon_events[size])
        print(f"  {size}-pixel SPC clusters: {spc_count}")

    # High-ADU Photon Hit Clusters
    print("\nHigh-ADU Photon Clusters (by assigned photon counts):")
    for size in range(1, 5):
        adu_count = photon_counts[size]
        print(f"  {size}-photon high-ADU clusters: {adu_count}")

    total_spc_hits = sum(len(photon_events[size]) for size in range(1, 5))
    total_high_adu_hits = sum(size * photon_counts[size] for size in range(1, 5))

    print(f"\nTotal SPC Photon Hits: {total_spc_hits}")
    print(f"Total High-ADU Photon Hits: {total_high_adu_hits}")
    print(f"Combined Total Photon Hits: {total_spc_hits + total_high_adu_hits}")

    # Visualizations
    bins = np.arange(0, np.max(pixel_values), 2)
    plot_full_histogram(pixel_values, mu_fit, sigma_fit, bins)
    plot_corrected_histogram(corrected_pixel_values, bins)
    plot_original_ccd_image(original_image)
    plot_corrected_ccd_image(corrected_image)
    plot_photon_hits(photon_events, image_index=image_index)
    plot_high_ADU_clusters(ccd_photon_hits, image_index=image_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPC engine with pedestal subtraction.")
    parser.add_argument("file_name", type=str, nargs="?", default=HDF5_FILE, help="Path to CCD HDF5 file.")
    parser.add_argument("--single_image", type=int, help="Process a single image index.")

    args = parser.parse_args()
    run_spc_engine(args.file_name, single_image=args.single_image)
