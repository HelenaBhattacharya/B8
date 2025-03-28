import numpy as np
import matplotlib.pyplot as plt
from config import CCD_SHAPE

def plot_photon_counts_vs_energy(bin_centers, hist_counts, title="Summed X-ray Spectrum (All Images)"):
    """Plots photon counts vs. energy."""
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist_counts, width=1, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Photon Count")
    plt.title(title)
    plt.grid(True)
    plt.xlim(bin_centers.min(), bin_centers.max())
    plt.show()

# def plot_solid_angle_adjusted_spectrum(bin_centers, corrected_intensity):
    #     """Plots spectrum after solid-angle correction."""
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(bin_centers, corrected_intensity, linestyle='-', linewidth=0.5, color='red', label="Solid-Angle Corrected")
    #     plt.xlabel("Energy (eV)")
    #     plt.ylabel("Photon Count (Corrected)")
    #     plt.title("Extracted X-ray Spectrum: Solid Angle Corrected")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.xlim(bin_centers.min(), bin_centers.max())
    #     plt.show()

def plot_solid_angle_adjusted_spectrum(bin_centers, corrected_intensity, corrected_errors):
    """Plots spectrum after solid-angle correction with Poisson error shading."""
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, corrected_intensity, linestyle='-', linewidth=0.1, color='blue',
             label="Solid Angle Corrected Spectrum")

    # Always show errors as shading
    plt.fill_between(bin_centers,
                     corrected_intensity - corrected_errors,
                     corrected_intensity + corrected_errors,
                     color='blue',
                     alpha=0.2,
                     label='Poisson Error')

    plt.xlabel("Energy (eV)")
    plt.ylabel("Photon Count (Corrected)")
    plt.title("Extracted X-ray Spectrum - Lineout (Solid Angle Adjusted)")
    plt.legend()
    plt.grid(True)
    plt.xlim(bin_centers.min(), bin_centers.max())
    plt.show()

# def plot_intensity_vs_energy(bin_centers, corrected_intensity):
#     """Plots intensity (Photon Count × Energy) vs. energy."""
#     intensity = bin_centers * corrected_intensity
#     plt.figure(figsize=(8, 6))
#     plt.plot(bin_centers, intensity, linestyle='-', linewidth=0.5, color='green',
#              label="Intensity (Photon Count × Energy)")
#     plt.xlabel("Energy (eV)")
#     plt.ylabel("Intensity")
#     plt.title("X-ray Spectrum: Intensity vs Energy")
#     plt.legend()
#     plt.grid(True)
#     plt.xlim(bin_centers.min(), bin_centers.max())
#     plt.show()

def plot_intensity_vs_energy(bin_centers, intensity, intensity_errors):
    """Intensity plot with Poisson errors shaded."""
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, intensity, linestyle='-', linewidth=0.3, color='black',
             label="Intensity (Photon Count × Energy)")

    plt.fill_between(bin_centers,
                     intensity - intensity_errors,
                     intensity + intensity_errors,
                     color='green',
                     alpha=0.3,
                     label='Poisson Error')

    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (Photon Count × Energy)")
    plt.title("X-ray Spectrum: Intensity vs Energy")
    plt.legend()
    plt.grid(True)
    plt.xlim(bin_centers.min(), bin_centers.max())
    plt.show()

def plot_log_intensity_vs_energy(bin_centers, corrected_intensity):
    """Plots log-scaled intensity vs. energy."""
    intensity = bin_centers * corrected_intensity
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, intensity, linestyle='-', linewidth=0.5, color='black', label="Log Intensity")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity")
    plt.yscale("log")
    plt.title("X-ray Spectrum: Log-Scaled Intensity")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.xlim(bin_centers.min(), bin_centers.max())
    plt.show()
#
# def plot_log_intensity_vs_energy(bin_centers, corrected_intensity, corrected_intensity_errors=None):
#     """Plots log-scaled intensity vs. energy with optional Poisson error shading."""
#     intensity = bin_centers * corrected_intensity
#
#     plt.figure(figsize=(8, 6))
#     plt.plot(bin_centers, intensity, linestyle='-', linewidth=0.5, color='black', label="Log Intensity")
#
#     # Optional shaded error band (only if errors are provided)
#     if corrected_intensity_errors is not None:
#         intensity_errors = corrected_intensity_errors * bin_centers
#
#         # Prevent negative values (log plot cannot handle them)
#         lower_bound = np.clip(intensity - intensity_errors, a_min=1e-10, a_max=None)
#         upper_bound = intensity + intensity_errors
#
#         plt.fill_between(bin_centers,
#                          lower_bound,
#                          upper_bound,
#                          color='grey',
#                          alpha=0.4,
#                          label='Poisson Error')
#
#     plt.xlabel("Energy (eV)")
#     plt.ylabel("Intensity")
#     plt.yscale("log")
#     plt.title("X-ray Spectrum: Log-Scaled Intensity")
#     plt.legend()
#     plt.grid(True, which="both", linestyle="--")
#     plt.xlim(bin_centers.min(), bin_centers.max())
#     plt.show()



def plot_ccd_hits(x_hits_combined, y_hits_combined):
    """Plots photon hits on CCD with (0,0) at top-left."""
    plt.figure(figsize=(8, 8))
    plt.scatter(x_hits_combined, y_hits_combined, color='cyan', s=0.1, alpha=0.7, label="Photon Hits")
    plt.xlabel("CCD X Position")
    plt.ylabel("CCD Y Position")
    plt.title("Photon Hits on CCD (All Images Combined)")
    plt.xlim(0, CCD_SHAPE[1])
    plt.ylim(CCD_SHAPE[0], 0)
    plt.legend(markerscale=4)
    plt.grid(False)
    plt.show()

def plot_normalized_spectrum(bin_centers, raw_counts, corrected_counts):
    """Plots normalized photon counts vs. energy (before and after correction)."""
    raw_norm = (raw_counts - np.min(raw_counts)) / (np.max(raw_counts) - np.min(raw_counts))
    corrected_norm = (corrected_counts - np.min(corrected_counts)) / (np.max(corrected_counts) - np.min(corrected_counts))

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, raw_norm, linestyle='-', linewidth=0.8, color='blue', label="Normalized Raw Spectrum")
    plt.plot(bin_centers, corrected_norm, linestyle='-', linewidth=0.8, color='red', label="Normalized Corrected Spectrum")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized Photon Count (a.u.)")
    plt.title("Normalized Photon Counts Before and After Solid Angle Correction")
    plt.legend()
    plt.grid(True)
    plt.xlim(bin_centers.min(), bin_centers.max())
    plt.show()

# Additional visualizations from original spectral reconstruction

def plot_extracted_spectrum_lineout(bin_centers, hist_counts):
    """Plots extracted X-ray spectrum lineout (ADU-weighted)."""
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, hist_counts, linestyle='-', linewidth=0.2, color='blue', label="Lineout Spectrum")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Photon Count")
    plt.title("Extracted X-ray Spectrum (SPC + High-ADU Clusters, Lineout)")
    plt.legend()
    plt.grid(True)
    plt.xlim(bin_centers.min(), bin_centers.max())
    plt.show()

def plot_summed_image(summed_image):
    """Plots summed CCD image for diagnostic purposes."""
    plt.figure(figsize=(8, 6))
    plt.imshow(summed_image, cmap='hot', origin='upper', extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
    plt.colorbar(label='Summed Pixel Intensity (ADU)')
    plt.title('Summed CCD Image (Diagnostic)')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

def plot_high_adu_positions(summed_image, high_adu_positions):
    """Plots high-ADU positions on summed CCD."""
    plt.figure(figsize=(8, 6))
    plt.imshow(summed_image, cmap='hot', origin='upper', extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
    if high_adu_positions:
        y_coords, x_coords = zip(*high_adu_positions)
        plt.scatter(x_coords, y_coords, color='cyan', s=5, label="High-ADU Pixels")
    plt.colorbar(label='Summed Pixel Intensity (ADU)')
    plt.title('High-ADU Pixels on Summed CCD')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.legend()
    plt.show()

# Wrapper function for convenience
def plot_all_spectral_results(bin_centers, raw_counts, corrected_counts, x_hits, y_hits):
    """Convenience function to quickly visualize all key spectral results."""
    plot_photon_counts_vs_energy(bin_centers, raw_counts)
    plot_solid_angle_adjusted_spectrum(bin_centers, corrected_counts)
    plot_intensity_vs_energy(bin_centers, corrected_counts)
    plot_log_intensity_vs_energy(bin_centers, corrected_counts)
    plot_normalized_spectrum(bin_centers, raw_counts, corrected_counts)
    plot_ccd_hits(x_hits, y_hits)



