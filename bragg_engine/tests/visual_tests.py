import numpy as np
import matplotlib.pyplot as plt
from bragg_engine.solid_angle import compute_solid_angle
from bragg_engine.mapping import compute_energy_map, rotation_matrix
from bragg_engine.preprocessing import preprocess_image
from bragg_engine.load import get_ccd_image
from bragg_engine.curve_fitting import parametric_curve
import pytest
from config import (
    HDF5_FILE,
    OPTIMIZED_PARAMS_FILE,
    QUADRATIC_PARAMS_FILE,
    CCD_SHAPE,
    PIXEL_SIZE,
    CCD_CENTER_X,
    CCD_CENTER_Y, CURVE_CENTER_1188_INIT, CURVE_CENTER_1218_INIT, ENERGY_LEVELS,
)

@pytest.mark.visual
def visual_solid_angle_test():
    x_prime, y_prime = np.meshgrid(
        np.linspace(-5e-3, 5e-3, 200),
        np.linspace(-5e-3, 5e-3, 200)
    )
    D = 0.083
    theta_bar_B = 37.2
    R = rotation_matrix(0, 0, 0)
    Omega = compute_solid_angle(x_prime, y_prime, D, theta_bar_B, R)

    plt.figure(figsize=(8, 6))
    plt.imshow(Omega, extent=[
        x_prime.min() * 1e3, x_prime.max() * 1e3,
        y_prime.min() * 1e3, y_prime.max() * 1e3
    ], cmap='viridis', origin='lower')
    plt.colorbar(label='Solid Angle (sr)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('CCD Solid Angle Distribution')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

@pytest.mark.visual
def visual_energy_dispersion_test():
    optimized_params = np.load(str(OPTIMIZED_PARAMS_FILE), allow_pickle=True)
    E_ij, x_prime, y_prime = compute_energy_map(optimized_params)
    x_prime_reindexed = x_prime / PIXEL_SIZE + CCD_CENTER_X
    y_prime_reindexed = y_prime / PIXEL_SIZE + CCD_CENTER_Y

    raw_image = get_ccd_image(HDF5_FILE, image_index=8)
    processed_image = preprocess_image(raw_image)

    plt.figure(figsize=(8, 6))
    plt.imshow(E_ij, cmap="jet", origin="upper", extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
    plt.colorbar(label="Energy (eV)")
    plt.title("Energy Mapping Across CCD Pixels")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(processed_image, cmap="hot", origin="upper", extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
    plt.colorbar(label="Normalized Intensity")
    plt.title("CCD Image with Isoenergy Contours")

    for level, color in zip([1188, 1218.5], ['cyan', 'white']):
        plt.contour(x_prime_reindexed, y_prime_reindexed, E_ij, levels=[level], colors=color, linewidths=0.5)

    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.legend(["1188 eV", "1218.5 eV"])
    plt.show()

@pytest.mark.visual
def visual_optimized_vs_experimental_test():
    optimized_params = np.load(str(OPTIMIZED_PARAMS_FILE), allow_pickle=True)
    quadratic_params = np.load(str(QUADRATIC_PARAMS_FILE), allow_pickle=True).item()

    E_ij, x_prime, y_prime = compute_energy_map(optimized_params)
    x_prime_reindexed = x_prime / PIXEL_SIZE + CCD_CENTER_X
    y_prime_reindexed = y_prime / PIXEL_SIZE + CCD_CENTER_Y

    y_exp = np.linspace(0, CCD_SHAPE[0], 500)
    x_center_1188 = CURVE_CENTER_1188_INIT
    x_center_1218 = CURVE_CENTER_1218_INIT

    x_exp_1188, y_exp_1188 = parametric_curve(y_exp, *quadratic_params["1188eV"][:2], x_center_1188, CCD_CENTER_Y)
    x_exp_1218, y_exp_1218 = parametric_curve(y_exp, *quadratic_params["1218.5eV"][:2], x_center_1218, CCD_CENTER_Y)

    plt.figure(figsize=(8, 6))
    plt.imshow(np.zeros(CCD_SHAPE), cmap='gray', origin='upper', extent=[0, CCD_SHAPE[1], CCD_SHAPE[0], 0])
    plt.title("Comparison of Theoretical and Experimental Isoenergy Fits")

    # Corrected line:
    for level, color in zip(ENERGY_LEVELS, ['cyan', 'white']):
        plt.contour(x_prime_reindexed, y_prime_reindexed, E_ij, levels=[level], colors=color, linewidths=1.0)

    plt.plot(x_exp_1188, y_exp_1188, 'r--', linewidth=1.0, label="Exp. 1188 eV")
    plt.plot(x_exp_1218, y_exp_1218, 'b--', linewidth=1.0, label="Exp. 1218.5 eV")

    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Optional manual runner
if __name__ == "__main__":
    print("Running all visual tests...\n")
    visual_solid_angle_test()
    visual_energy_dispersion_test()
    visual_optimized_vs_experimental_test()
    print("✅ All visual tests completed.")