# intensity_error.py
import numpy as np

def poisson_error(photon_counts):
    """
    Compute Poisson statistical error (standard deviation) for given photon counts.

    Args:
        photon_counts (np.ndarray): Array of photon counts per energy bin.

    Returns:
        np.ndarray: Poisson error per bin (sqrt(N)).
    """
    return np.sqrt(photon_counts)