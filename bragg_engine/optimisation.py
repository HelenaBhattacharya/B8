import numpy as np
import cma  # CMA-ES optimizer
from scipy.interpolate import interp1d
from bragg_engine.mapping import compute_energy_map
from bragg_engine.curve_fitting import parametric_curve
from config import ENERGY_LEVELS, PIXEL_SIZE, PARAM_BOUNDS, INITIAL_PARAMS,CCD_CENTER_X, CCD_CENTER_Y, CURVE_CENTER_1218_INIT, CURVE_CENTER_1188_INIT, CCD_SHAPE

def load_experimental_fits():
    """
    Loads experimental quadratic fits from stored coefficients and computes x, y values.
    """
    quadratic_params = np.load("quadratic_params.npy", allow_pickle=True).item()
    a_1188, b_1188, _ = quadratic_params["1188eV"]
    a_1218, b_1218, _ = quadratic_params["1218.5eV"]

    # Ensure y_exp matches the old version exactly
    y_exp = np.linspace(0, CCD_SHAPE[0] - 1, 500)
    x_exp_1188, y_exp_1188 = parametric_curve(y_exp, a_1188, b_1188, CURVE_CENTER_1188_INIT, CCD_CENTER_Y)
    x_exp_1218, y_exp_1218 = parametric_curve(y_exp, a_1218, b_1218, CURVE_CENTER_1218_INIT, CCD_CENTER_Y)

    return (x_exp_1188, y_exp_1188), (x_exp_1218, y_exp_1218)

def interpolate_contour(x_contour, y_contour, num_samples=500):
    """
    Interpolates extracted contour points to match experimental fit.
    """
    if len(x_contour) < 2 or len(y_contour) < 2:
        return np.array([]), np.array([])  # Prevent failure in optimization

    unique_indices = np.unique(y_contour, return_index=True)[1]
    y_sorted = y_contour[unique_indices]
    x_sorted = x_contour[unique_indices]

    f_interp = interp1d(y_sorted, x_sorted, kind="linear", fill_value="extrapolate")
    y_interp = np.linspace(y_sorted.min(), y_sorted.max(), num_samples)
    x_interp = f_interp(y_interp)

    return x_interp, y_interp

def loss_function(params, x_exp_1188, y_exp_1188, x_exp_1218, y_exp_1218):
    """
    Computes the mean squared error (MSE) between experimental quadratic fits and theoretical contours.
    """
    E_ij, x_prime, y_prime = compute_energy_map(params)

    # Convert to CCD-centered coordinates
    x_prime_reindexed = x_prime / PIXEL_SIZE + CCD_CENTER_X
    y_prime_reindexed = y_prime / PIXEL_SIZE + CCD_CENTER_Y

    contour_points = {}
    for energy in ENERGY_LEVELS:
        mask = np.abs(E_ij - energy) < 0.1
        iso_x = x_prime_reindexed[mask]
        iso_y = y_prime_reindexed[mask]

        if len(iso_x) < 1 or len(iso_y) < 1:
            return np.inf  # Avoid bad solutions

        contour_points[energy] = (iso_x, iso_y)

    # Interpolate extracted contour points
    x_contour_1188, y_contour_1188 = interpolate_contour(contour_points[1188][0], contour_points[1188][1])
    x_contour_1218, y_contour_1218 = interpolate_contour(contour_points[1218.5][0], contour_points[1218.5][1])

    # Compute MSE Loss
    L_1188 = np.mean((x_contour_1188 - x_exp_1188) ** 2 + (y_contour_1188 - y_exp_1188) ** 2)
    L_1218 = np.mean((x_contour_1218 - x_exp_1218) ** 2 + (y_contour_1218 - y_exp_1218) ** 2)

    return L_1188 + L_1218


def optimize_parameters(save_path="optimized_params.npy"):
    """
    Runs the CMA-ES optimization algorithm and saves the optimized parameters.
    """
    (x_exp_1188, y_exp_1188), (x_exp_1218, y_exp_1218) = load_experimental_fits()

    es = cma.CMAEvolutionStrategy(INITIAL_PARAMS, 0.02, {
        'bounds': list(map(list, zip(*PARAM_BOUNDS))),
        'tolfun': 1e-12,
        'tolx': 1e-12,
    })
    es.optimize(lambda params: loss_function(params, x_exp_1188, y_exp_1188, x_exp_1218, y_exp_1218))
    optimized_params = es.result.xbest

    np.save(save_path, np.array(optimized_params, dtype=np.float64))
    return optimized_params

def compute_optimized_energy_map():
    """
    Computes the energy map using the optimized parameters.
    """
    optimized_params = np.load("optimized_params.npy", allow_pickle=True)
    E_ij_opt, x_prime, y_prime = compute_energy_map(optimized_params)

    # No need to re-create meshgrid; return values directly
    return E_ij_opt, x_prime / PIXEL_SIZE + CCD_CENTER_X, y_prime / PIXEL_SIZE + CCD_CENTER_Y