import numpy as np
import matplotlib.pyplot as plt


def plot_full_histogram(pixel_values, mu_fit, sigma_fit, bins):
    """Plot the full pixel intensity histogram with Gaussian fit before pedestal subtraction."""
    hist, bin_edges = np.histogram(pixel_values, bins=bins)

    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], hist, width=2, alpha=0.8, color="royalblue", edgecolor="black", label="Full CCD Histogram")

    # Overlay Gaussian Fit
    x_fit = np.linspace(0, 100, 500)
    gaussian_fit = np.exp(-((x_fit - mu_fit) ** 2) / (2 * sigma_fit ** 2))
    gaussian_fit *= np.max(hist) / np.max(gaussian_fit)  # Normalize height
    plt.plot(x_fit, gaussian_fit, 'r-', label=f'Gaussian Fit (Pedestal): μ={mu_fit:.2f}, σ={sigma_fit:.2f}')

    # Vertical line for pedestal mean
    plt.axvline(mu_fit, color='red', linestyle='--', label="Fitted Pedestal Mean")

    plt.yscale('log')
    plt.xlabel("Pixel Signal (ADU)")
    plt.ylabel("Number of Counts")
    plt.title("Full Pixel Intensity Histogram with Improved Readability")
    plt.legend()
    plt.grid(False)
    plt.show()


def plot_corrected_histogram(corrected_pixel_values, bins):
    """Plot the corrected pixel intensity histogram after pedestal subtraction."""
    hist_corrected, bin_edges = np.histogram(corrected_pixel_values, bins=bins)

    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], hist_corrected, width=2, alpha=0.8, color="darkorange", edgecolor="black",
            label="Corrected Pixel Histogram")
    plt.axvline(0, color='black', linestyle='--', label="Zero ADU Baseline")

    plt.yscale('log')
    plt.xlabel("Pixel Signal (ADU) After Pedestal Subtraction")
    plt.ylabel("Number of Counts")
    plt.title("Corrected Pixel Intensity Distribution After Pedestal Removal")
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_original_ccd_image(image):
    """Plot the original CCD image before pedestal removal."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='hot', origin='upper')
    plt.colorbar(label='Pixel Intensity (ADU)')
    plt.title('Original CCD Image Before Pedestal Removal')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

def plot_corrected_ccd_image(corrected_image):
    """Plot the corrected CCD image after pedestal removal."""
    plt.figure(figsize=(8, 6))
    plt.imshow(corrected_image, cmap='hot', origin='upper')
    plt.colorbar(label='Pixel Intensity (ADU) After Pedestal Subtraction')
    plt.title('Corrected CCD Image After Pedestal Removal')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

import matplotlib.pyplot as plt

def plot_photon_hits(photon_events, image_index=None):
    """
    Plots all detected photon hits on a blank CCD, using different colors for cluster sizes.

    Args:
        photon_events (dict): Dictionary containing photon hits for cluster sizes 1 to 4.
        image_index (int, optional): Index of the image being processed.
    """
    colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red'}
    labels = {1: "1-Pixel Hits", 2: "2-Pixel Hits", 3: "3-Pixel Hits", 4: "4-Pixel Hits"}

    plt.figure(figsize=(8, 8))

    for cluster_size in range(1, 5):
        if photon_events[cluster_size]:  # Check if there are any events
            x_vals, y_vals, _ = zip(*photon_events[cluster_size])
            plt.scatter(y_vals, x_vals, color=colors[cluster_size], s=0.5, alpha=0.7, label=labels[cluster_size])

    plt.xlabel("CCD X Position")
    plt.ylabel("CCD Y Position")
    plt.gca().invert_yaxis()  # Keep the CCD coordinate system consistent
    plt.title(f"Photon Hits on CCD - Image {image_index}" if image_index is not None else "Photon Hits on CCD")
    plt.legend(markerscale=5)
    plt.show()


def plot_high_ADU_clusters(ccd_photon_hits, image_index=None):
    """
    Plots the photon hit map after charge redistribution for high ADU clusters.

    Args:
        ccd_photon_hits (numpy.ndarray): 2D array marking assigned photon hit positions.
        image_index (int, optional): Image index for labeling.
    """
    hit_x, hit_y = np.where(ccd_photon_hits > 0)  # Get positions of photon hits

    plt.figure(figsize=(8, 8))
    plt.scatter(hit_y, hit_x, color='blue', s=5, label="Photon Hits")
    plt.gca().invert_yaxis()  # Ensure CCD coordinate consistency
    plt.xlabel("CCD X Position")
    plt.ylabel("CCD Y Position")
    plt.title(f"Photon Hits After Charge Redistribution - Image {image_index}" if image_index is not None else "Photon Hits After Charge Redistribution")
    plt.legend()
    plt.show()
