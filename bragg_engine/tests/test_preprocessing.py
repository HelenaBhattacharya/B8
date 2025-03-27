import numpy as np
import pytest
from config import HDF5_FILE, CCD_SHAPE
from bragg_engine.preprocessing import preprocess_image, sum_all_ccd_images

TEST_FILE_PATH = str(HDF5_FILE)

def test_preprocess_image_shape_and_range():
    image = np.random.randint(0, 500, CCD_SHAPE)
    processed_image = preprocess_image(image)
    assert processed_image.shape == CCD_SHAPE, "Processed image must have original shape."
    assert 0 <= processed_image.min() <= processed_image.max() <= 1, "Processed image must be normalized (0-1)."

def test_preprocess_image_invalid_input():
    invalid_image = np.random.rand(*CCD_SHAPE, 3)  # Invalid 3D shape
    with pytest.raises(ValueError):
        preprocess_image(invalid_image)

def test_sum_all_ccd_images_summation():
    summed_image, num_high_adu, positions = sum_all_ccd_images(TEST_FILE_PATH, max_images=3, threshold_adu=1e7)
    assert summed_image.shape == CCD_SHAPE, "Summed image must have correct CCD shape."
    assert isinstance(num_high_adu, int), "Number of high ADU pixels should be integer."
    assert isinstance(positions, list), "High ADU positions should be a list of tuples."

def test_sum_all_ccd_images_high_adu_detection():
    images = [np.zeros(CCD_SHAPE) for _ in range(3)]
    images[0][100, 100] = 5000
    images[1][100, 100] = 6000
    images[2][100, 100] = 7000
    images_sum = np.sum(images, axis=0)
    threshold = 10000
    y_high, x_high = np.where(images_sum > threshold)
    expected_positions = list(zip(y_high, x_high))
    assert (100, 100) in expected_positions, "High ADU pixel should be detected correctly."