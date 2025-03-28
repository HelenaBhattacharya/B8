import numpy as np
import pytest
from bragg_engine.curve_fitting import (
    parametric_curve,
    extract_peak_x,
    fit_quadratic_curve
)
from config import CCD_SHAPE, CCD_CENTER_X, CCD_CENTER_Y

# Mock data setup (simple synthetic Gaussian peaks)
@pytest.fixture
def mock_image():
    img = np.zeros(CCD_SHAPE)
    x_center, y_center = CCD_CENTER_X, CCD_CENTER_Y

    # Create synthetic peaks along a known quadratic curve
    for y in range(100, 1900, 100):
        x = int(5e-5 * (y - y_center)**2 + 0.01 * (y - y_center) + x_center)
        img[y-5:y+5, x-3:x+3] = 100  # Small rectangular peak region

    # Add slight noise
    img += np.random.normal(0, 1, img.shape)
    return img

def test_parametric_curve():
    x_center, y_center = CCD_CENTER_X, CCD_CENTER_Y
    y = np.array([1024, 1100, 1200])
    a, b = 5e-5, 1e-2
    x, y_out = parametric_curve(y, a, b, x_center, y_center)

    # Check correct output shapes
    assert x.shape == y.shape, "X and Y output shapes do not match."

    # Check curve passes exactly through the center point at y_center
    x_central, _ = parametric_curve(y_center, a, b, x_center, y_center)
    assert np.isclose(x_central, x_center), "Curve does not pass through center."

def test_extract_peak_x(mock_image):
    x_center, y_center = CCD_CENTER_X, CCD_CENTER_Y
    x_coords, y_coords, sigma_values = extract_peak_x(mock_image, x_center, y_center, num_sections=18)

    # Check correct shapes
    assert len(x_coords) == len(y_coords), "Mismatch in x and y coordinate lengths."
    assert len(sigma_values) == len(y_coords), "Mismatch in sigma and y coordinate lengths."

    # Check no NaNs in the smoothed x coordinates
    assert not np.isnan(x_coords).any(), "NaNs found in smoothed x coordinates."

    # Check sigma values are reasonable
    assert np.nanmean(sigma_values) > 0, "Average sigma should be positive."

def test_fit_quadratic_curve(mock_image, tmp_path):
    x_center, y_center = CCD_CENTER_X, CCD_CENTER_Y
    save_path = tmp_path / "sigma_values.npy"

    coeffs = fit_quadratic_curve(mock_image, x_center, y_center, save_sigma_path=str(save_path))

    # Check if coefficients have correct length (quadratic -> 3 coefficients)
    assert len(coeffs) == 3, "Quadratic fit must return exactly 3 coefficients."

    # Check sigma values file was saved correctly
    assert save_path.exists(), "Sigma values file was not saved."

    sigma_values = np.load(save_path)
    assert len(sigma_values) > 0, "Saved sigma values file is empty."