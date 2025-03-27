import numpy as np
from scipy.ndimage import maximum_filter, generic_filter
from bragg_engine.load import get_ccd_image, load_ccd_images
from config import THRESHOLD_PERCENTILE, HIGH_ADU_THRESHOLD, MAX_IMAGES

def vertical_pooling(image, pool_shape=(5, 3)):
    """Applies max-pooling over vertical strips to retain the brightest pixel."""
    return maximum_filter(image, size=pool_shape)

def remove_large_clusters(image, y_threshold=220, cluster_threshold=10):
    """Removes large bright clusters in the bottom region while keeping small structures."""
    def cluster_logic(values):
        center = values[4]  # Center pixel
        bright_count = np.sum(values > 0)
        return 0 if bright_count > cluster_threshold else center  # Suppress large clusters

    image[:y_threshold, :] = generic_filter(image[:y_threshold, :], cluster_logic, size=(3, 3))
    return image

def preprocess_image(image, threshold_percentile=THRESHOLD_PERCENTILE):
    """
    Apply preprocessing steps: log-scaling, normalization, vertical pooling, and noise filtering.

    Args:
        image (np.ndarray): Input CCD image.
        threshold_percentile (float): Percentile threshold for intensity filtering.

    Returns:
        np.ndarray: Processed image.
    """
    if image.ndim != 2:
        raise ValueError(f"ERROR: preprocess_image expected a 2D image, but got shape {image.shape}")

    log_image = np.log1p(image)
    log_image = (log_image - np.min(log_image)) / (np.max(log_image) - np.min(log_image))
    log_image = vertical_pooling(log_image, pool_shape=(5, 3))
    threshold = np.percentile(log_image, threshold_percentile)
    log_image = np.where(log_image >= threshold, log_image, 0)
    log_image = remove_large_clusters(log_image, y_threshold=220, cluster_threshold=10)

    return log_image

def sum_all_ccd_images(file_path, max_images=MAX_IMAGES, threshold_adu=HIGH_ADU_THRESHOLD):
    """
    Sums all CCD images pixel by pixel and finds pixels exceeding a given ADU threshold.

    Args:
        file_path (str): Path to the HDF5 file.
        max_images (int): Number of images to sum.
        threshold_adu (int): ADU value to detect high-intensity pixels.

    Returns:
        np.ndarray: 2D summed CCD image.
        int: Number of high ADU pixels.
        list[tuple]: List of (y, x) positions of pixels exceeding threshold.
    """
    images = load_ccd_images(file_path, max_images=max_images)

    # Sum all images pixel-wise
    summed_image = np.sum(images, axis=0)

    # Identify high ADU pixels
    y_high, x_high = np.where(summed_image > threshold_adu)
    high_adu_positions = list(zip(y_high, x_high))

    return summed_image, len(high_adu_positions), high_adu_positions
