import numpy as np
import matplotlib.pyplot as plt
from config import PIXEL_SIZE, CCD_SHAPE, CCD_CENTER_X, CCD_CENTER_Y, CURVE_CENTER_1188_INIT,CURVE_CENTER_1218_INIT, ENERGY_LEVELS
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
        plt.imshow(np.log1p(image), cmap=cmap, origin='upper', extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
    else:
        plt.imshow(image, cmap=cmap, origin='upper', extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
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
    plt.imshow(image, cmap='hot', origin='upper', extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
    plt.colorbar(label='Normalized Intensity')

    # Define y-range
    y_values = np.linspace(0, image.shape[0] - 1, 1000)
    y_relative = y_values - CCD_CENTER_Y

    # Compute experimental quadratic fits
    x_fit_1188_exp = coeffs_1188_exp[0] * y_relative ** 2 + coeffs_1188_exp[1] * y_relative + CURVE_CENTER_1188_INIT
    x_fit_1218_exp = coeffs_1218_exp[0] * y_relative ** 2 + coeffs_1218_exp[1] * y_relative + CURVE_CENTER_1218_INIT

    # Plot experimental fits
    plt.plot(x_fit_1188_exp, y_values, 'lime', linewidth=0.5, linestyle='dashed', label="Exp. 1188 eV")
    plt.plot(x_fit_1218_exp, y_values, 'blue', linewidth=0.5, linestyle='dashed', label="Exp. 1218.5 eV")

    plt.title("Experimental Quadratic Fits on CCD Image")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.legend()
    plt.show()

def plot_energy_map(image, energy_map, x_prime, y_prime, energy_levels=ENERGY_LEVELS):
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
    extent = [0, CCD_SHAPE[1], CCD_SHAPE[0], 0]
    im = plt.imshow(image, cmap='hot', origin='upper', extent=extent)
    cbar = plt.colorbar(im)
    cbar.set_label("Normalised Intensity", fontsize=13)
    cbar.ax.tick_params(labelsize=13)
    colors = ['cyan', 'white']
    contour_labels = []
    for level, color in zip(energy_levels, colors):
        contour = plt.contour(x_prime, y_prime, energy_map, levels=[level], colors=color, linewidths=0.5)
        if contour.allsegs[0]:  # Only add label if the contour exists
            contour_labels.append(f"{level} eV")
    plt.xlabel("Pixel X", fontsize=13)
    plt.ylabel("Pixel Y", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if contour_labels:
        plt.legend(contour_labels, fontsize=13)
    plt.title("CCD Image with Isoenergy Contours", fontsize=13)

    plt.tight_layout()
    plt.show()


def plot_curve_comparison(quadratic_params_file, optimized_quadratic_params_file):
    quadratic_params = np.load(quadratic_params_file, allow_pickle=True).item()
    optimized_quadratic_params = np.load(optimized_quadratic_params_file, allow_pickle=True)

    # Extract parameters
    a_1188, b_1188, c_1188 = quadratic_params["1188eV"]
    a_1218, b_1218, c_1218 = quadratic_params["1218.5eV"]
    a_opt, b_opt, c_opt = optimized_quadratic_params

    y_values = np.linspace(0, CCD_SHAPE[0] - 1, 500)
    x_opt, _ = parametric_curve(y_values, a_opt, b_opt, c_opt, CCD_CENTER_Y)

    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros(CCD_SHAPE), cmap='gray', extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
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
    im = plt.imshow(summed_image, cmap='hot', origin='upper')
    cbar = plt.colorbar(im)
    cbar.set_label("Summed ADU Intensity", fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel("Pixel X", fontsize=15)
    plt.ylabel("Pixel Y", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.title("Summed CCD Image (20 Exposures)")

    plt.tight_layout()
    plt.show()