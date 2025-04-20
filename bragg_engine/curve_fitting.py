import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from bragg_engine.preprocessing import preprocess_image
from bragg_engine.load import get_ccd_image
from config import CCD_CENTER_X, CCD_CENTER_Y

# Define Quadratic Parametric Curve Function
def parametric_curve(y, a, b, x_center, y_center):
    """
    Defines a quadratic parametric curve centered at (x_center, y_center).
    """
    x = a * (y - y_center) ** 2 + b * (y - y_center)
    return x + x_center, y


def extract_peak_x(image, x_center, y_center, num_sections=20):
    """
    Extracts x-coordinates of peak intensities in horizontal slices along y.
    Uses Gaussian fitting to refine the peak location.

    Returns:
        np.ndarray: Smoothed peak x-coordinates.
        np.ndarray: Corresponding y-coordinates.
        np.ndarray: Extracted sigma values (Gaussian width).
    """
    base_band_width = 30
    scaling_factor = 0.025
    extra_band_width_bottom = 50
    y_min, y_max = 0, image.shape[0] - 1
    y_sections = np.linspace(y_min, y_max, num_sections + 1)

    optimized_x_coords = []
    optimized_y_coords = y_sections[:-1]  # Store the corresponding y-values
    sigma_values = []  # Store extracted Gaussian sigma values

    for i in range(len(y_sections) - 1):
        y_start, y_end = int(y_sections[i]), int(y_sections[i + 1])
        x_mid = parametric_curve(y_sections[i], 5.0000e-05, 1.0000e-02, x_center, y_center)[0]

        # Dynamically adjust band width based on y-position
        if y_end > 1800:
            band_width = base_band_width + scaling_factor * abs(y_end - y_center) + extra_band_width_bottom - 10
        elif y_end < 500:
            band_width = base_band_width + scaling_factor * abs(y_end - y_center)
        else:
            band_width = base_band_width

        band_width = int(band_width)
        x_start, x_end = int(x_mid - band_width // 2), int(x_mid + band_width // 2)

        # Extract intensity profile in the band
        if 0 <= x_start and x_end < image.shape[1]:
            section_intensity = np.sum(image[y_start:y_end, x_start:x_end], axis=0)
        else:
            section_intensity = np.zeros(band_width)

        x_pixels = np.linspace(-band_width // 2, band_width // 2, len(section_intensity))

        try:
            sigma_guess = 2 if y_end < 1800 else 3.5
            popt, _ = curve_fit(
                lambda x, I_max, x0, sigma: I_max * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)),
                x_pixels, section_intensity, p0=[np.max(section_intensity), 0, sigma_guess]
            )
            optimized_x = x_mid + popt[1]  # Adjust x-mid based on Gaussian peak
            sigma_values.append(popt[2])  # Extract fitted sigma
            optimized_x_coords.append(optimized_x)
        except RuntimeError:
            optimized_x_coords.append(x_mid)  # If fit fails, use initial estimate
            sigma_values.append(np.nan)  # Store NaN for missing values

    optimized_x_coords = np.array(optimized_x_coords)
    optimized_y_coords = np.array(optimized_y_coords)
    sigma_values = np.array(sigma_values)  # Convert sigma list to NumPy array

    # Apply smoothing filter
    optimized_x_coords_smoothed = savgol_filter(optimized_x_coords, window_length=5, polyorder=2)

    return optimized_x_coords_smoothed, optimized_y_coords, sigma_values


def fit_quadratic_curve(image, x_center, y_center, save_sigma_path=None):
    """
    Fits a quadratic curve to extracted peak positions and saves Gaussian sigma values.

    Args:
        image (np.ndarray): CCD image.
        x_center (int): Expected x-center of spectral feature.
        y_center (int): Expected y-center of spectral feature.
        save_sigma_path (str, optional): Path to save fitted sigma values.

    Returns:
        np.ndarray: Quadratic fit coefficients [a, b, c].
    """
    optimized_x_coords, optimized_y_coords, sigma_values = extract_peak_x(image, x_center, y_center)

    # Convert to CCD-centered coordinates
    S_exp = np.column_stack((optimized_x_coords - CCD_CENTER_X, optimized_y_coords - CCD_CENTER_Y))

    # Perform quadratic fitting
    coeffs, residuals, _, _, _ = np.polyfit(S_exp[:, 1], S_exp[:, 0], 2, full=True)

    # **Ensure sigma values are saved
    if save_sigma_path:
        np.save(save_sigma_path, sigma_values)

    return coeffs
