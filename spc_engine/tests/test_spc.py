import numpy as np
import pytest
from spc_engine.spc import detect_photon_events

@pytest.fixture
def synthetic_ccd_image():
    image = np.zeros((100, 100), dtype=np.float32)

    # Single-pixel photon hit at (10, 10)
    image[10, 10] = 125  # above secondary threshold (ADU_sp - sigma_N = 110.5)

    # Two-pixel horizontal photon cluster at (20,20) and (20,21)
    image[20, 20] = 70
    image[20, 21] = 60  # sum = 130 (>110.5, <200)

    # Three-pixel L-shape photon cluster at (30,30), (31,30), (31,31)
    image[30, 30] = 40
    image[31, 30] = 45
    image[31, 31] = 50  # sum = 135 (>110.5, <200)

    # Four-pixel square photon cluster at (40,40), (40,41), (41,40), (41,41)
    image[40, 40] = 35
    image[40, 41] = 30
    image[41, 40] = 40
    image[41, 41] = 45  # sum = 150 (>110.5, <200)

    return image

def test_detect_photon_events(synthetic_ccd_image):
    results = detect_photon_events(synthetic_ccd_image, sigma_N=9.5, ADU_sp=120)

    adu_weighted_ccd_final, cluster_pixel_map, photon_events, high_ADU_clusters, filtered_high_ADU_clusters, thresholds = results

    T_initial, T_secondary, ADU_sp = thresholds

    # Threshold checks
    assert np.isclose(T_initial, 14.25), f"Incorrect initial threshold: {T_initial}"
    assert np.isclose(T_secondary, 110.5), f"Incorrect secondary threshold: {T_secondary}"
    assert ADU_sp == 120, f"Incorrect ADU_sp: {ADU_sp}"

    # Single-pixel detection
    assert len(photon_events[1]) == 1, f"Expected 1 single-pixel photon hit, got {len(photon_events[1])}"
    assert photon_events[1][0][:2] == (10, 10), f"Single-pixel photon incorrect position: {photon_events[1][0][:2]}"

    # Two-pixel detection
    assert len(photon_events[2]) == 1, f"Expected 1 two-pixel photon cluster, got {len(photon_events[2])}"
    assert photon_events[2][0][:2] in [(20, 20), (20, 21)], f"Two-pixel photon incorrect position: {photon_events[2][0][:2]}"

    # Three-pixel detection
    assert len(photon_events[3]) == 1, f"Expected 1 three-pixel photon cluster, got {len(photon_events[3])}"
    assert photon_events[3][0][:2] in [(30, 30), (31, 30), (31, 31)], f"Three-pixel photon incorrect position: {photon_events[3][0][:2]}"

    # Four-pixel detection (2×2 square only valid shape)
    assert len(photon_events[4]) == 1, f"Expected 1 four-pixel photon cluster, got {len(photon_events[4])}"
    assert photon_events[4][0][:2] in [(40, 40), (40, 41), (41, 40), (41, 41)], f"Four-pixel photon incorrect position: {photon_events[4][0][:2]}"

    # Ensure no unexpected high-ADU clusters (since none should exceed 200 ADU here)
    assert len(high_ADU_clusters) == 0, f"Expected 0 high ADU clusters, got {len(high_ADU_clusters)}"