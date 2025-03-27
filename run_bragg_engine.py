import os
import numpy as np
from bragg_engine.load import get_ccd_image
from bragg_engine.preprocessing import preprocess_image, sum_all_ccd_images
from bragg_engine.curve_fitting import fit_quadratic_curve
from bragg_engine.optimisation import optimize_parameters, compute_optimized_energy_map
from bragg_engine.visualisation import plot_image, plot_fitted_curves, plot_energy_map, plot_summed_image
from bragg_engine.energy_resolution import (
    extract_optimized_params,
    calc_total_energy_resolution,
    compute_energy_dispersion
)
from config import (
    HDF5_FILE,
    CCD_CENTER_Y,
    ENERGY_LEVELS,
    OPTIMIZED_PARAMS_FILE,
    QUADRATIC_PARAMS_FILE,
    SPOT_AREA_M2
)
# import matplotlib
#
# matplotlib.use('TkAgg')


def main(force_recompute=False, file_name=None):
    if file_name is None:
        file_name = HDF5_FILE  # default from config.py
        image_index = 8

    # === Define known x-centers for spectral lines ===
    x_center_1188 = 1424
    x_center_1218 = 1284

    # Load and preprocess image
    raw_image = get_ccd_image(file_name, image_index)
    processed_image = preprocess_image(raw_image)

    # === Fit experimental curves and save ===
    coeffs_1188 = fit_quadratic_curve(processed_image, x_center_1188, CCD_CENTER_Y,
                                      save_sigma_path="fitted_sigma_1188.npy")
    coeffs_1218 = fit_quadratic_curve(processed_image, x_center_1218, CCD_CENTER_Y,
                                      save_sigma_path="fitted_sigma_1218.npy")
    quadratic_params = {"1188eV": coeffs_1188, "1218.5eV": coeffs_1218}
    np.save(str(QUADRATIC_PARAMS_FILE), quadratic_params, allow_pickle=True)

    if not os.path.exists("fitted_sigma_1188.npy") or not os.path.exists("fitted_sigma_1218.npy"):
        raise FileNotFoundError("ERROR: Sigma files were not saved correctly!")

    print("\nExperimental Quadratic Parameters:")
    for label, coeffs in quadratic_params.items():
        print(f"{label}: a = {coeffs[0]:.12f}, b = {coeffs[1]:.12f}, c = {coeffs[2]:.12f}")

    # === Run optimization ===
    if not OPTIMIZED_PARAMS_FILE.exists() or force_recompute:
        print("\nRunning optimization...")
        optimized_params = optimize_parameters(save_path=str(OPTIMIZED_PARAMS_FILE))
    else:
        print("\nLoading precomputed optimized parameters...")
        optimized_params = np.load(str(OPTIMIZED_PARAMS_FILE), allow_pickle=True)

    print("\nOptimized Bragg Parameters:")
    print(f"D = {optimized_params[0]:.12f} m")
    print(f"θ̄_B = {optimized_params[1]:.12f}°")
    print(f"α_x = {optimized_params[2]:.12f}°")
    print(f"α_y = {optimized_params[3]:.12f}°")

    # === Compute optimized energy map ===
    E_ij_opt, x_prime_opt, y_prime_opt = compute_optimized_energy_map()

    # === Reload fitted parameters ===
    quadratic_params = np.load(str(QUADRATIC_PARAMS_FILE), allow_pickle=True).item()
    coeffs_1188_exp = quadratic_params["1188eV"]
    coeffs_1218_exp = quadratic_params["1218.5eV"]

    # === Compute energy resolution ===
    print("\n🔬 Computing Energy Resolution...")
    sigma_values_1188 = np.load("fitted_sigma_1188.npy")
    sigma_values_1218 = np.load("fitted_sigma_1218.npy")

    sqrt_2ln2 = 2 * np.sqrt(2 * np.log(2))
    mean_fwhm_1188 = np.nanmean(sqrt_2ln2 * sigma_values_1188)
    mean_fwhm_1218 = np.nanmean(sqrt_2ln2 * sigma_values_1218)

    dE_dx_1188 = compute_energy_dispersion(E_ij_opt, x_prime_opt, ENERGY_LEVELS[0])
    dE_dx_1218 = compute_energy_dispersion(E_ij_opt, x_prime_opt, ENERGY_LEVELS[1])

    if np.isnan(dE_dx_1188) or np.isnan(dE_dx_1218):
        print("\n⚠️ ERROR: Invalid dE/dx values. Skipping resolution calc.")
    else:
        res_1188 = calc_total_energy_resolution(
            optimized_params[1], optimized_params[0], mean_fwhm_1188, dE_dx_1188, SPOT_AREA_M2
        )
        res_1218 = calc_total_energy_resolution(
            optimized_params[1], optimized_params[0], mean_fwhm_1218, dE_dx_1218, SPOT_AREA_M2
        )

        print("\nEnergy Resolution Contributions (in eV):")
        print(f"1188 eV: Pixel={res_1188['pixel_broadening']:.3f}, "
              f"Source={res_1188['source_broadening']:.3f}, "
              f"Rocking={res_1188['rocking_curve_broadening']:.3f}, "
              f"Total={res_1188['total_broadening']:.3f}")
        print(f"1218.5 eV: Pixel={res_1218['pixel_broadening']:.3f}, "
              f"Source={res_1218['source_broadening']:.3f}, "
              f"Rocking={res_1218['rocking_curve_broadening']:.3f}, "
              f"Total={res_1218['total_broadening']:.3f}")

        print("\nRecommended Bin Width (Rounded Up):")
        print(f"1188 eV: {np.ceil(res_1188['total_broadening']):.0f} eV")
        print(f"1218.5 eV: {np.ceil(res_1218['total_broadening']):.0f} eV")

    # === Visualization ===
    plot_image(raw_image, "Raw Image", cmap="viridis")
    plot_image(processed_image, "Preprocessed CCD Image", cmap='hot')
    plot_fitted_curves(processed_image, coeffs_1188_exp, coeffs_1218_exp)
    plot_energy_map(processed_image, E_ij_opt, x_prime_opt, y_prime_opt)

    # === Summed CCD image ===
    summed_image, num_high_adu, high_adu_positions = sum_all_ccd_images(file_name)
    print(f"\nHigh ADU Pixels: {num_high_adu}")
    print(f"Sample: {high_adu_positions[:10]}")
    plot_summed_image(summed_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Bragg engine with optional optimization recalculation.")
    parser.add_argument("file_name", nargs="?", help="Path to HDF5 file (optional, default from config.py)")
    parser.add_argument("--force-recompute", action="store_true", help="Force recalculation of optimized parameters.")
    args = parser.parse_args()

    main(force_recompute=args.force_recompute, file_name=args.file_name)


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Run Bragg engine with optional optimization recalculation.")
#     parser.add_argument("--force-recompute", action="store_true")
#     args = parser.parse_args()
#     main(force_recompute=args.force_recompute)

# import os
# import numpy as np
# from bragg_engine.load import get_ccd_image
# from bragg_engine.preprocessing import preprocess_image, sum_all_ccd_images
# from bragg_engine.curve_fitting import fit_quadratic_curve
# from bragg_engine.optimisation import optimize_parameters, compute_optimized_energy_map
# from bragg_engine.visualisation import plot_image, plot_fitted_curves, plot_energy_map, plot_summed_image
# from bragg_engine.energy_resolution import (
#     extract_optimized_params, calc_total_energy_resolution, compute_energy_dispersion
# )
# from config import HDF5_FILE, CCD_CENTER_Y, ENERGY_LEVELS
# import matplotlib
# matplotlib.use('TkAgg')
#
#
# def main(force_recompute=False):
#     file_name = HDF5_FILE
#     image_index = 8
#
#     # === Define known x-centers for spectral lines ===
#     x_center_1188 = 1424
#     x_center_1218 = 1284
#
#     # Load and preprocess image
#     raw_image = get_ccd_image(file_name, image_index)
#     processed_image = preprocess_image(raw_image)
#
#     # === Fit experimental curves and save ===
#     coeffs_1188 = fit_quadratic_curve(processed_image, x_center_1188, CCD_CENTER_Y, save_sigma_path="fitted_sigma_1188.npy")
#     coeffs_1218 = fit_quadratic_curve(processed_image, x_center_1218, CCD_CENTER_Y, save_sigma_path="fitted_sigma_1218.npy")
#     quadratic_params = {"1188eV": coeffs_1188, "1218.5eV": coeffs_1218}
#     np.save("quadratic_params.npy", quadratic_params, allow_pickle=True)
#
#     if not os.path.exists("fitted_sigma_1188.npy") or not os.path.exists("fitted_sigma_1218.npy"):
#         raise FileNotFoundError("ERROR: Sigma files were not saved correctly!")
#
#     print("\nExperimental Quadratic Parameters:")
#     for label, coeffs in quadratic_params.items():
#         print(f"{label}: a = {coeffs[0]:.12f}, b = {coeffs[1]:.12f}, c = {coeffs[2]:.12f}")
#
#     # === Run optimization ===
#     if not os.path.exists("optimized_params.npy") or force_recompute:
#         print("\nRunning optimization...")
#         optimized_params = optimize_parameters(save_path="optimized_params.npy")
#     else:
#         print("\nLoading precomputed optimized parameters...")
#         optimized_params = np.load("optimized_params.npy", allow_pickle=True)
#
#     print("\nOptimized Bragg Parameters:")
#     print(f"D = {optimized_params[0]:.12f} m")
#     print(f"θ̄_B = {optimized_params[1]:.12f}°")
#     print(f"α_x = {optimized_params[2]:.12f}°")
#     print(f"α_y = {optimized_params[3]:.12f}°")
#
#     # === Compute optimized energy map ===
#     E_ij_opt, x_prime_opt, y_prime_opt = compute_optimized_energy_map()
#
#     # === Reload fitted parameters ===
#     quadratic_params = np.load("quadratic_params.npy", allow_pickle=True).item()
#     coeffs_1188_exp = quadratic_params["1188eV"]
#     coeffs_1218_exp = quadratic_params["1218.5eV"]
#
#     # === Compute energy resolution ===
#     print("\n🔬 Computing Energy Resolution...")
#     sigma_values_1188 = np.load("fitted_sigma_1188.npy")
#     sigma_values_1218 = np.load("fitted_sigma_1218.npy")
#
#     sqrt_2ln2 = 2 * np.sqrt(2 * np.log(2))
#     mean_fwhm_1188 = np.nanmean(sqrt_2ln2 * sigma_values_1188)
#     mean_fwhm_1218 = np.nanmean(sqrt_2ln2 * sigma_values_1218)
#
#     dE_dx_1188 = compute_energy_dispersion(E_ij_opt, x_prime_opt, ENERGY_LEVELS[0])
#     dE_dx_1218 = compute_energy_dispersion(E_ij_opt, x_prime_opt, ENERGY_LEVELS[1])
#
#     if np.isnan(dE_dx_1188) or np.isnan(dE_dx_1218):
#         print("\n ERROR: Invalid dE/dx values. Skipping resolution calc.")
#     else:
#         spot_area_m2 = 28.5e-12
#         res_1188 = calc_total_energy_resolution(optimized_params[1], optimized_params[0], mean_fwhm_1188, dE_dx_1188, spot_area_m2)
#         res_1218 = calc_total_energy_resolution(optimized_params[1], optimized_params[0], mean_fwhm_1218, dE_dx_1218, spot_area_m2)
#
#         print("\nEnergy Resolution Contributions (in eV):")
#         print(f"1188 eV: Pixel={res_1188['pixel_broadening']:.3f}, Source={res_1188['source_broadening']:.3f}, "
#               f"Rocking={res_1188['rocking_curve_broadening']:.3f}, Total={res_1188['total_broadening']:.3f}")
#         print(f"1218.5 eV: Pixel={res_1218['pixel_broadening']:.3f}, Source={res_1218['source_broadening']:.3f}, "
#               f"Rocking={res_1218['rocking_curve_broadening']:.3f}, Total={res_1218['total_broadening']:.3f}")
#
#         print("\nRecommended Bin Width (Rounded Up):")
#         print(f"1188 eV: {np.ceil(res_1188['total_broadening']):.0f} eV")
#         print(f"1218.5 eV: {np.ceil(res_1218['total_broadening']):.0f} eV")
#
#     # === Visualization ===
#     plot_image(raw_image, "Raw Image", cmap="viridis")
#     plot_image(processed_image, "Preprocessed CCD Image", cmap='hot')
#     plot_fitted_curves(processed_image, coeffs_1188_exp, coeffs_1218_exp)
#     plot_energy_map(processed_image, E_ij_opt, x_prime_opt, y_prime_opt)
#
#     summed_image, num_high_adu, high_adu_positions = sum_all_ccd_images(file_name)
#     print(f"\nHigh ADU Pixels: {num_high_adu}")
#     print(f"Sample: {high_adu_positions[:10]}")
#     plot_summed_image(summed_image)
#
#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Run Bragg engine with optional optimization recalculation.")
#     parser.add_argument("--force-recompute", action="store_true")
#     args = parser.parse_args()
#     main(force_recompute=args.force_recompute)