import pytest
import numpy as np
import os
from bragg_engine.load import load_ccd_images, get_ccd_image
from config import HDF5_FILE, CCD_SHAPE
TEST_FILE_PATH = str(HDF5_FILE)

def test_load_ccd_images_count():
    images = load_ccd_images(TEST_FILE_PATH, max_images=5)
    assert len(images) == 5

def test_load_ccd_images_no_images():
    with pytest.raises(FileNotFoundError):
        load_ccd_images("invalid/path/file.h5")

def test_get_ccd_image_valid_index():
    image = get_ccd_image(TEST_FILE_PATH, image_index=2)
    assert image.shape == CCD_SHAPE

def test_get_ccd_image_invalid_index():
    with pytest.raises(IndexError):
        get_ccd_image(TEST_FILE_PATH, image_index=100)