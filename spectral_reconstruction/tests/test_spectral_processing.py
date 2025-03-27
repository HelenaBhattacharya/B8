import numpy as np

from spectral_reconstruction.spectral_processing import (
remove_high_ADU_pixels,
extract_photon_hits,
resolve_overlaps,
map_photon_energies,
sum_photon_hits,
solid_angle_correction)
import pytest

def test_remove_high_ADU_pixels():
    adu_weighted_ccd = np.ones((10, 10))
    ccd_redistributed = np.ones((10, 10))
    high_adu_positions = [(5, 5)]

    remove_high_ADU_pixels(adu_weighted_ccd, ccd_redistributed, high_adu_positions)

    # Ensure the central region is zeroed
    assert np.count_nonzero(adu_weighted_ccd[3:7, 3:7]) == 0
    assert np.count_nonzero(ccd_redistributed[3:7, 3:7]) == 0


def test_extract_photon_hits():
        cluster_pixel_map = np.zeros((5, 5))
        adu_weighted_ccd = np.zeros((5, 5))
        ccd_redistributed = np.zeros((5, 5))

        cluster_pixel_map[1, 1] = 1
        adu_weighted_ccd[1, 1] = 150

        ccd_redistributed[3, 3] = 250

        spc_hits, high_adu_hits = extract_photon_hits(cluster_pixel_map, adu_weighted_ccd, ccd_redistributed)

        assert len(spc_hits) == 1
        assert np.array_equal(spc_hits[0], [1, 1, 150])

        assert len(high_adu_hits) == 1
        assert np.array_equal(high_adu_hits[0], [3, 3, 250])


def test_resolve_overlaps():
    spc_hits = np.array([[2, 2, 150], [3, 3, 120]])
    high_adu_hits = np.array([[3, 3, 300]])

    clean_spc, clean_high_adu, num_removed = resolve_overlaps(spc_hits, high_adu_hits)

    assert num_removed == 1
    assert len(clean_spc) == 1
    assert np.array_equal(clean_spc[0], [2, 2, 150])
    assert len(clean_high_adu) == 1  # unchanged


def test_map_photon_energies(mocker):
    mocker.patch("spectral_reconstruction.spectral_processing.compute_energy_map",
                 return_value=(np.array([[100, 200], [300, 400]]), None, None))

    photon_hits = np.array([[0, 0, 5], [1, 1, 10]])
    optimized_params = (None,)  # parameters aren't needed since function is mocked
    energies, adus = map_photon_energies(photon_hits, optimized_params)

    assert np.array_equal(energies, [100, 400])
    assert np.array_equal(adus, [5, 10])
    

def test_sum_photon_hits():
    photon_energies = np.array([100, 150])
    photon_adus = np.array([2, 3])
    energy_bins = np.array([50, 125, 200])

    hist_counts = sum_photon_hits(photon_energies, photon_adus, energy_bins)

    assert np.array_equal(hist_counts, [2, 3])

def test_solid_angle_correction():
        hist_counts = np.array([10, 20])
        energy_bins = np.array([0, 100, 200])
        E_ij = np.array([[50, 150]])
        Omega_ij = np.array([[2, 4]])

        corrected = solid_angle_correction(hist_counts, energy_bins, E_ij, Omega_ij)

        assert np.allclose(corrected, [5.0, 5.0])