import numpy as np
import h5py
import itertools
from config import MAX_IMAGES

def load_ccd_images(file_path, max_images=MAX_IMAGES):
    """
    Load multiple CCD images from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        max_images (int): Maximum number of images to load.

    Returns:
        list[np.ndarray]: List of CCD images.
    """
    image_data = []
    with h5py.File(file_path, 'r') as datafile:
        for i in itertools.islice(itertools.count(start=0), max_images):
            dataset_path = f'Configure:0000/Run:0000/CalibCycle:{i:04d}/Princeton::FrameV2/SxrEndstation.0:Princeton.0/data'
            d = datafile.get(dataset_path)
            if d is not None:
                image_data.append(d[0])  # Extract first index as the image
            else:
                break

    if not image_data:
        raise ValueError(f"No CCD images found in file: {file_path}")

    return image_data

def get_ccd_image(file_path, image_index=8):
    """
    Retrieve a specific CCD image from the loaded dataset.

    Args:
        file_path (str): Path to the HDF5 file.
        image_index (int): Index of the image to extract.

    Returns:
        np.ndarray: The selected CCD image.
    """
    images = load_ccd_images(file_path)
    if image_index >= len(images):
        raise IndexError(f"Image index {image_index} out of range. Found {len(images)} images.")
    return images[image_index]