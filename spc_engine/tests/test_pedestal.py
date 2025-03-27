import numpy as np
import pytest
from spc_engine.pedestal import compute_pedestal, gaussian, process_pedestal_correction

@pytest.fixture
def synthetic_gaussian():
    mu, sigma = 50, 10
    return np.random.normal(mu, sigma, size=(2048, 2048)).astype(np.float32)

def test_gaussian_function():
    x = np.array([0, 1, 2])
    assert np.allclose(gaussian(x, 1, 1, 1), np.exp(-0.5 * (x - 1)**2))

def test_compute_pedestal_ideal(synthetic_gaussian):
    mu_fit, sigma_fit = compute_pedestal(synthetic_gaussian)
    assert np.isclose(mu_fit, 50, atol=1.5), f"mu_fit={mu_fit} not close enough to 50"
    assert np.isclose(sigma_fit, 10, atol=1.0), f"sigma_fit={sigma_fit} not close enough to 10"

def test_process_single_image(tmp_path):
    image = np.random.normal(30, 5, size=(2048, 2048)).astype(np.float32)
    hdf5_path = tmp_path / "test.h5"

    import h5py
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset(
            'Configure:0000/Run:0000/CalibCycle:0000/Princeton::FrameV2/SxrEndstation.0:Princeton.0/data',
            data=image[np.newaxis, :, :]
        )

    mu_fit, sigma_fit = process_pedestal_correction(str(hdf5_path), image_index=0)
    assert np.isclose(mu_fit, 30, atol=1.5), f"mu_fit={mu_fit} not close enough to 30"
    assert np.isclose(sigma_fit, 5, atol=1.0), f"sigma_fit={sigma_fit} not close enough to 5"
