import pytest
import numpy as np
from bragg_engine.energy_resolution import (
    extract_optimized_params,
    compute_energy_dispersion,
    calc_pixel_broadening,
    calc_source_broadening,
    calc_rocking_curve_broadening,
    calc_total_energy_resolution,
)
from bragg_engine.mapping import compute_energy_map, pixel_size
from config import CCD_SHAPE

# Mock values for testing
TEST_ENERGY_LEVEL = 1200  # eV, example energy level
SPOT_AREA_M2 = 1e-8  # Example spot area in m² for testing purposes
SIGMA_PIXELS = 2.5  # Example fitted sigma in pixels
MOCK_PARAMS = [0.083, 37.2, -1.89, 55.8]  # Move this to global scope if not already

@pytest.fixture
def optimized_params(tmp_path):
    path = tmp_path / "optimized_params.npy"
    np.save(path, MOCK_PARAMS)
    return MOCK_PARAMS

def test_compute_energy_dispersion(optimized_params):
    E_ij, x_prime, _ = compute_energy_map(MOCK_PARAMS)
    dE_dx = compute_energy_dispersion(E_ij, x_prime, TEST_ENERGY_LEVEL)
    assert np.isfinite(dE_dx), "Energy dispersion must be finite."
    assert dE_dx != 0, "Energy dispersion should not be zero."


def test_calc_pixel_broadening():
    broadening = calc_pixel_broadening(37.2, 0.083)
    assert broadening > 0, "Pixel broadening should be positive."


def test_calc_source_broadening():
    broadening = calc_source_broadening(37.2, 0.083, SPOT_AREA_M2)
    assert broadening > 0, "Source broadening should be positive."


def test_calc_rocking_curve_broadening():
    dE_dx_pixel = 5000  # example dispersion value in eV/m
    broadening = calc_rocking_curve_broadening(SIGMA_PIXELS, dE_dx_pixel)
    assert broadening > 0, "Rocking curve broadening should be positive."


def test_calc_total_energy_resolution():
    theta_B_deg = 37.2
    D = 0.083
    dE_dx_pixel = 5000

    resolutions = calc_total_energy_resolution(
        theta_B_deg, D, SIGMA_PIXELS, dE_dx_pixel, SPOT_AREA_M2
    )

    assert resolutions["pixel_broadening"] > 0, "Pixel broadening should be positive."
    assert resolutions["source_broadening"] > 0, "Source broadening should be positive."
    assert resolutions["rocking_curve_broadening"] > 0, "Rocking curve broadening should be positive."
    assert resolutions["total_broadening"] > 0, "Total broadening should be positive."
    assert resolutions["total_broadening"] >= max(
        resolutions["pixel_broadening"],
        resolutions["source_broadening"],
        resolutions["rocking_curve_broadening"]
    ), "Total broadening must be the largest."


def test_compute_energy_dispersion_invalid():
    E_ij = np.full((CCD_SHAPE), np.nan)  # Invalid energy map
    x_prime = np.zeros(CCD_SHAPE)
    with pytest.raises(ValueError):
        compute_energy_dispersion(E_ij, x_prime, TEST_ENERGY_LEVEL)