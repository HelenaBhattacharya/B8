import numpy as np
import h5py
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from config import (
    HDF5_FILE,
    MAX_IMAGES,
    PEDESTAL_BIN_WIDTH,
    PEDESTAL_FIT_RANGE,
    PEDESTAL_SIGMA_INIT
)

def load_ccd_images(file_name, num_images=1):
    """Load CCD images from an HDF5 file."""
    with h5py.File(file_name, 'r') as datafile:
        image_data = []
        for i in itertools.count(start=0):
            d = datafile.get(
                f'Configure:0000/Run:0000/CalibCycle:{i:04d}/Princeton::FrameV2/SxrEndstation.0:Princeton.0/data')
            if d is not None:
                image_data.append(d[0])
            else:
                break
    return image_data[:num_images]

def gaussian(x, A, mu, sigma):
    """Gaussian function for pedestal fitting."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def compute_pedestal(image):
    """Perform Gaussian fit on the pedestal and return fitted parameters."""
    pixel_values = image.flatten()
    bins = np.arange(0, np.max(pixel_values), PEDESTAL_BIN_WIDTH)
    hist, bin_edges = np.histogram(pixel_values, bins=bins)

    # Fit only the pedestal region
    A_init = np.max(hist)
    mu_init = bin_edges[np.argmax(hist)]
    sigma_init = PEDESTAL_SIGMA_INIT

    fit_range = (bin_edges[:-1] >= PEDESTAL_FIT_RANGE[0]) & (bin_edges[:-1] <= PEDESTAL_FIT_RANGE[1])
    x_fit = bin_edges[:-1][fit_range]
    y_fit = hist[fit_range]

    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[A_init, mu_init, sigma_init])
        A_fit, mu_fit, sigma_fit = popt
    except RuntimeError:
        mu_fit, sigma_fit = mu_init, sigma_init

    return mu_fit, sigma_fit

def process_pedestal_correction(file_name=None, image_index=0):
    """
    Load and process a single CCD image for pedestal subtraction.

    Args:
        file_name (str): HDF5 file path (defaults to config)
        image_index (int): Index of the image to process

    Returns:
        tuple: (mu_fit, sigma_fit)
    """
    file_name = file_name or HDF5_FILE
    images = load_ccd_images(file_name, num_images=image_index + 1)
    image = images[image_index]

    return compute_pedestal(image)