import numpy as np
import matplotlib.pyplot as plt
from config import PIXEL_SIZE
from bragg_engine.curve_fitting import parametric_curve  # Import quadratic fit function
from bragg_engine.curve_fitting import parametric_curve

def plot_image(image, title, cmap='viridis', log_scale=False):
    """
    Plots a CCD image with matrix-style (upper origin).

    Args:
        image (np.ndarray): Image data to plot.
        title (str): Plot title.
        cmap (str): Colormap.
        log_scale (bool): Apply logarithmic scale to intensity.
    """
    plt.figure(figsize=(8, 6))
    if log_scale:
        plt.imshow(np.log1p(image), cmap=cmap, origin='upper', extent=[0, 2048, 2048, 0])
    else:
        plt.imshow(image, cmap=cmap, origin='upper', extent=[0, 2048, 2048, 0])
    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

def plot_fitted_curves(image, coeffs_1188_exp, coeffs_1218_exp):
    """
    Plots experimental quadratic fits overlaid on the CCD image.

    Args:
        image (np.ndarray): CCD image.
        coeffs_1188_exp (tuple): Experimental fit coefficients for 1188 eV.
        coeffs_1218_exp (tuple): Experimental fit coefficients for 1218.5 eV.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='hot', origin='upper', extent=[0, 2048, 2048, 0])
    plt.colorbar(label='Normalized Intensity')

    # Define y-range
    y_values = np.linspace(0, image.shape[0] - 1, 1000)
    y_relative = y_values - 1024  # Fixing Center-to-Upper Origin Issue

    # Compute experimental quadratic fits
    x_fit_1188_exp = coeffs_1188_exp[0] * y_relative ** 2 + coeffs_1188_exp[1] * y_relative + 1424
    x_fit_1218_exp = coeffs_1218_exp[0] * y_relative ** 2 + coeffs_1218_exp[1] * y_relative + 1284

    # Plot experimental fits
    plt.plot(x_fit_1188_exp, y_values, 'lime', linewidth=0.5, linestyle='dashed', label="Exp. 1188 eV")
    plt.plot(x_fit_1218_exp, y_values, 'blue', linewidth=0.5, linestyle='dashed', label="Exp. 1218.5 eV")

    plt.title("Experimental Quadratic Fits on CCD Image")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.legend()
    plt.show()

def plot_energy_map(image, energy_map, x_prime, y_prime, energy_levels=[1188, 1218.5]):
    """
    Overlays theoretical isoenergy curves from the energy mapping onto the CCD image.

    Args:
        image (np.ndarray): Preprocessed CCD image.
        energy_map (np.ndarray): Computed energy map.
        x_prime (np.ndarray): X-coordinate grid from compute_energy_map().
        y_prime (np.ndarray): Y-coordinate grid from compute_energy_map().
        energy_levels (list): List of energy values to contour.
    """
    plt.figure(figsize=(8, 6))

    # Ensure the CCD image is displayed in correct pixel space (0 to 2048)
    extent = [0, 2048, 2048, 0]

    # Plot the preprocessed CCD image
    plt.imshow(image, cmap='hot', origin='upper', extent=extent)
    plt.colorbar(label="Normalized Intensity")
    plt.title("CCD Image with Isoenergy Contours")

    # Overlay isoenergy contours at correct positions
    colors = ['cyan', 'white']
    contour_labels = []
    for level, color in zip(energy_levels, colors):
        contour = plt.contour(x_prime, y_prime, energy_map, levels=[level], colors=color, linewidths=0.5)
        if contour.allsegs[0]:
            contour_labels.append(f"{level} eV")

    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.legend(contour_labels)
    plt.show()


def plot_curve_comparison(quadratic_params_file, optimized_quadratic_params_file, ccd_shape=(2048, 2048)):
    """
    Load the quadratic and optimized parameters and plot their corresponding curves on a blank CCD.
    """
    quadratic_params = np.load(quadratic_params_file, allow_pickle=True).item()
    optimized_quadratic_params = np.load(optimized_quadratic_params_file, allow_pickle=True)

    # Extract parameters
    a_1188, b_1188, c_1188 = quadratic_params["1188eV"]
    a_1218, b_1218, c_1218 = quadratic_params["1218.5eV"]
    a_opt, b_opt, c_opt = optimized_quadratic_params  # Now correctly extracting 3 values

    y_values = np.linspace(0, ccd_shape[0] - 1, 500)
    x_opt, _ = parametric_curve(y_values, a_opt, b_opt, c_opt, 1024)

    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros(ccd_shape), cmap='gray', extent=[0, ccd_shape[1], ccd_shape[0], 0])
    plt.plot(x_opt, y_values, label="Optimized Fit", color='cyan', linestyle='-', linewidth=1.5)

    plt.xlabel("CCD X (pixels)")
    plt.ylabel("CCD Y (pixels)")
    plt.title("Quadratic vs. Optimized Fit Comparison")
    plt.legend()
    plt.show()

def plot_summed_image(summed_image):
    """
    Plots the summed CCD image from multiple exposures.

    Args:
        summed_image (np.ndarray): The pixel-wise summed CCD image.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(summed_image, cmap='hot', origin='upper')
    plt.colorbar(label="Summed ADU Intensity")
    plt.title("Summed CCD Image (20 Exposures)")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()
