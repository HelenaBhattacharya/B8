import numpy as np
import pytest
from config import (
    OPTIMIZED_PARAMS_FILE,
    QUADRATIC_PARAMS_FILE,
    ENERGY_LEVELS,
    CCD_CENTER_Y,
    CCD_SHAPE,
    CURVE_CENTER_1218_INIT,
    CURVE_CENTER_1188_INIT
)
from bragg_engine.mapping import compute_energy_map
from bragg_engine.optimisation import (
    interpolate_contour,
    loss_function,
)
from bragg_engine.curve_fitting import parametric_curve


@pytest.fixture
def optimized_params():
    return np.load(str(OPTIMIZED_PARAMS_FILE), allow_pickle=True)


@pytest.fixture
def quadratic_params():
    return np.load(str(QUADRATIC_PARAMS_FILE), allow_pickle=True).item()


@pytest.fixture
def experimental_fits(quadratic_params):
    y_exp = np.linspace(0, CCD_SHAPE[0] - 1, 500)

    # Fixed x-centers used in run_bragg_engine
    x_center_1188 = CURVE_CENTER_1188_INIT
    x_center_1218 = CURVE_CENTER_1218_INIT

    x_exp_1188, y_exp_1188 = parametric_curve(
        y_exp, *quadratic_params["1188eV"][:2], x_center_1188, CCD_CENTER_Y
    )
    x_exp_1218, y_exp_1218 = parametric_curve(
        y_exp, *quadratic_params["1218.5eV"][:2], x_center_1218, CCD_CENTER_Y
    )
    return (x_exp_1188, y_exp_1188), (x_exp_1218, y_exp_1218)


def test_load_experimental_fits(experimental_fits):
    (x_1188, y_1188), (x_1218, y_1218) = experimental_fits
    assert len(x_1188) == len(y_1188) == 500
    assert len(x_1218) == len(y_1218) == 500


def test_interpolate_contour():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    x_interp, y_interp = interpolate_contour(x, y, num_samples=10)
    assert len(x_interp) == 10
    assert len(y_interp) == 10


def test_interpolate_contour_insufficient_points():
    x, y = np.array([1]), np.array([1])
    x_interp, y_interp = interpolate_contour(x, y)
    assert len(x_interp) == 0 and len(y_interp) == 0


def test_loss_function_returns_finite(experimental_fits):
    (x_1188, y_1188), (x_1218, y_1218) = experimental_fits
    params = [0.083, 37.2, -1.89, 55.8]
    loss = loss_function(params, x_1188, y_1188, x_1218, y_1218)
    assert np.isfinite(loss)


def test_loss_function_invalid_params(experimental_fits):
    (x_1188, y_1188), (x_1218, y_1218) = experimental_fits
    params = [0.0, 0.0, 0.0, 0.0]
    loss = loss_function(params, x_1188, y_1188, x_1218, y_1218)
    assert loss == np.inf


def test_compute_optimized_energy_map(optimized_params):
    E_ij_opt, x_prime, y_prime = compute_energy_map(optimized_params)
    assert E_ij_opt.shape == CCD_SHAPE

