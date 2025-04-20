# spc.py
import numpy as np
import pickle
import pandas as pd
import os
from config import (
    SPC_SIGMA_N, SPC_ADU_SINGLE_PHOTON, SPC_FANO_FACTOR,
    SPC_CLUSTER_SUM_MAX, SPC_HIGH_ADU_PIXEL_THRESHOLD
)

def detect_photon_events(
    ccd_image_corrected,
    sigma_N=SPC_SIGMA_N,
    ADU_sp=SPC_ADU_SINGLE_PHOTON,
    f_Fano=SPC_FANO_FACTOR
):
    """
    Detects single-photon events and high-ADU clusters from a pedestal-subtracted CCD image.

    Applies two ADU thresholds to classify pixel clusters into single-, multi-pixel SPC events
    (1–4 pixels), or high-ADU clusters. Cluster shapes are matched using a rotated template approach.

    Args:
        ccd_image_corrected (np.ndarray): Pedestal-subtracted CCD image.
        sigma_N (float): Noise standard deviation used for initial thresholding.
        ADU_sp (float): Approximate ADU value of a single photon.
        f_Fano (float): Fano factor (currently unused, reserved for future use).

    Returns:
        tuple: (
            adu_weighted_ccd_final (np.ndarray): CCD map with ADU assigned to brightest cluster pixel,
            cluster_pixel_map (np.ndarray): Map with cluster sizes (0 if not a photon hit),
            photon_events (dict): Detected SPC events grouped by cluster size {1–4},
            high_ADU_clusters (list): All irregular or saturated clusters,
            filtered_high_ADU_clusters (list): Subset of clusters passing pixel-level ADU threshold,
            thresholds (tuple): (T_initial, T_secondary, ADU_sp)
        )
    """
    # Step 1: Threshold calculation
    T_initial = 1.5 * sigma_N
    T_secondary = ADU_sp - sigma_N

    # Step 2: Apply first threshold to remove noise
    thresholded_image = np.where(ccd_image_corrected >= T_initial, ccd_image_corrected, 0)

    # Step 3: Define Single-Photon Cluster Shapes
    cluster_shapes = {
        2: [np.array([[1, 1]]), np.array([[1], [1]])],
        3: [
            np.array([[1, 1, 0], [0, 1, 0]]), np.array([[1, 0], [1, 1]]),
            np.array([[1, 1], [1, 0]]), np.array([[1, 1], [0, 1]])
        ],
        4: [np.array([[1, 1], [1, 1]])]
    }

    # Step 4: Initialize Storage
    photon_events = {1: [], 2: [], 3: [], 4: []}
    adu_weighted_ccd = np.zeros_like(ccd_image_corrected, dtype=np.float32)
    cluster_pixel_map = np.zeros_like(ccd_image_corrected, dtype=np.int8)
    used_pixels = np.zeros_like(ccd_image_corrected, dtype=bool)
    high_ADU_clusters = []

    # Step 5.1: Detect single-pixel photon hits
    for i in range(1, thresholded_image.shape[0] - 1):
        for j in range(1, thresholded_image.shape[1] - 1):
            if used_pixels[i, j]:
                continue

            pixel_value = thresholded_image[i, j]
            if pixel_value > 0:
                region = thresholded_image[i - 1:i + 2, j - 1:j + 2]
                neighbors = np.delete(region.flatten(), 4)

                if pixel_value >= T_secondary and np.all(neighbors < T_initial):
                    photon_events[1].append((i, j, pixel_value))
                    adu_weighted_ccd[i, j] = pixel_value
                    cluster_pixel_map[i, j] = 1
                    used_pixels[i, j] = True

    # Step 5.2: Detect multi-pixel clusters (4→3→2)
    for num_pixels in [4, 3, 2]:
        for i in range(1, thresholded_image.shape[0] - 1):
            for j in range(1, thresholded_image.shape[1] - 1):
                if used_pixels[i, j]:
                    continue

                region = thresholded_image[i - 1:i + 2, j - 1:j + 2]
                shape_matched = False

                for shape in cluster_shapes[num_pixels]:
                    for rotation in range(4):
                        rotated_shape = np.rot90(shape, rotation)
                        shape_h, shape_w = rotated_shape.shape

                        if shape_h > 3 or shape_w > 3:
                            continue

                        sub_region = region[:shape_h, :shape_w]
                        if sub_region.shape != rotated_shape.shape:
                            continue

                        if np.all((sub_region > 0) == (rotated_shape > 0)):
                            cluster_pixels = np.argwhere(rotated_shape > 0) + [i - 1, j - 1]

                            if any(used_pixels[x, y] for x, y in cluster_pixels):
                                continue

                            cluster_sum = np.sum(sub_region[rotated_shape > 0])
                            brightest_pixel = max(cluster_pixels, key=lambda p: thresholded_image[p[0], p[1]])

                            if T_secondary <= cluster_sum <= SPC_CLUSTER_SUM_MAX:
                                photon_events[num_pixels].append((brightest_pixel[0], brightest_pixel[1], cluster_sum))
                                adu_weighted_ccd[brightest_pixel[0], brightest_pixel[1]] = cluster_sum
                                cluster_pixel_map[brightest_pixel[0], brightest_pixel[1]] = num_pixels
                                shape_matched = True

                            for x, y in cluster_pixels:
                                used_pixels[x, y] = True

                            break

                # Step 5.3: Handle high ADU or irregular shapes
                if not shape_matched:
                    cluster_pixels = np.argwhere(region > 0) + [i - 1, j - 1]
                    cluster_pixels = [(x, y, thresholded_image[x, y]) for x, y in cluster_pixels]
                    cluster_sum = np.sum(region[region > 0])

                    if cluster_sum > SPC_CLUSTER_SUM_MAX:
                        high_ADU_clusters.append({
                            "pixels": cluster_pixels,
                            "total_ADU": cluster_sum
                        })

    # Step 6: Apply second threshold
    adu_weighted_ccd_final = np.where(adu_weighted_ccd >= T_secondary, adu_weighted_ccd, 0)

    # Step 6.5: Filter high ADU clusters (pixel values ≥50)
    filtered_high_ADU_clusters = []
    for cluster in high_ADU_clusters:
        if all(thresholded_image[x, y] >= SPC_HIGH_ADU_PIXEL_THRESHOLD for x, y, _ in cluster["pixels"]):
            filtered_high_ADU_clusters.append(cluster)

    thresholds = (T_initial, T_secondary, ADU_sp)

    return (adu_weighted_ccd_final, cluster_pixel_map, photon_events, high_ADU_clusters,
            filtered_high_ADU_clusters, thresholds)


def save_spc_results(adu_map, cluster_pixel_map, high_ADU_clusters, photon_events, image_index):
    """
    Saves SPC detection results to disk for a given CCD image index.

    Outputs:
    - ADU-weighted CCD map (`adu_weighted_ccd_final.npy`)
    - Cluster size map (`cluster_pixel_map.npy`)
    - High ADU clusters (`high_ADU_clusters.pkl`)
    - Single-photon hit data table (`photon_hits_single_photon_clusters.csv`)

    Args:
        adu_map (np.ndarray): Final ADU-weighted CCD output from detection.
        cluster_pixel_map (np.ndarray): Map of SPC cluster sizes.
        high_ADU_clusters (list): Raw high-ADU cluster information.
        photon_events (dict): Dictionary of photon hit lists by cluster size.
        image_index (int): Image index for naming the output folder.
    """
    folder = str(image_index)
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, "adu_weighted_ccd_final.npy"), adu_map)
    np.save(os.path.join(folder, "cluster_pixel_map.npy"), cluster_pixel_map)

    with open(os.path.join(folder, "high_ADU_clusters.pkl"), "wb") as f:
        pickle.dump(high_ADU_clusters, f)

    photon_data = []
    for n in range(1, 5):
        for (x, y, adu) in photon_events[n]:
            photon_data.append([x, y, n, adu])
    df = pd.DataFrame(photon_data, columns=["X", "Y", "Cluster Size", "ADU Value"])
    df.to_csv(os.path.join(folder, "photon_hits_single_photon_clusters.csv"), index=False)
