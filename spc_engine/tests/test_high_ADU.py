import numpy as np
import pytest
import pickle
from spc_engine.high_ADU import (
    apply_ADU_threshold,
    redistribute_charge
)

@pytest.fixture
def synthetic_high_ADU_clusters():
    # Define synthetic clusters mimicking realistic high-ADU cases
    clusters = [
        {
            "pixels": [(10, 10, 250), (10, 11, 100), (11, 10, 80)],  # total = 430 (4 photons)
            "total_ADU": 430
        },
        {
            "pixels": [(20, 20, 150), (20, 21, 80)],  # total = 230 (2 photons)
            "total_ADU": 230
        },
        {
            "pixels": [(30, 30, 60), (30, 31, 50)],  # total = 110 (1 photon)
            "total_ADU": 110
        },
        {
            "pixels": [(40, 40, 19), (40, 41, 18)],  # Below threshold, should be removed
            "total_ADU": 37
        }
    ]
    return clusters

def test_apply_ADU_threshold(synthetic_high_ADU_clusters):
    filtered_clusters = apply_ADU_threshold(synthetic_high_ADU_clusters, threshold_ADU=20)

    # First cluster: No pixels removed
    assert len(filtered_clusters[0]["pixels"]) == 3, "First cluster pixels incorrectly filtered"
    assert filtered_clusters[0]["total_ADU"] == 430, "Incorrect total ADU after threshold for cluster 1"

    # Second cluster: No pixels removed
    assert len(filtered_clusters[1]["pixels"]) == 2, "Second cluster pixels incorrectly filtered"
    assert filtered_clusters[1]["total_ADU"] == 230, "Incorrect total ADU after threshold for cluster 2"

    # Third cluster: Pixels >=20 ADU remain
    assert len(filtered_clusters[2]["pixels"]) == 2, "Third cluster pixels incorrectly filtered"
    assert filtered_clusters[2]["total_ADU"] == 110, "Incorrect total ADU after threshold for cluster 3"

    # Fourth cluster: Both pixels <20 ADU, cluster should be empty
    assert len(filtered_clusters[3]["pixels"]) == 0, "Fourth cluster pixels not correctly filtered out"
    assert filtered_clusters[3]["total_ADU"] == 0, "Fourth cluster total ADU should be zero after threshold"

def test_redistribute_charge(synthetic_high_ADU_clusters):
    filtered_clusters = apply_ADU_threshold(synthetic_high_ADU_clusters, threshold_ADU=20)
    ccd_thresholded, ccd_redistributed, ccd_photon_hits, num_photon_hits, photon_counts = redistribute_charge(filtered_clusters, ccd_size=(50, 50))

    # Check photon count assignment
    assert photon_counts[4] == 1, f"Expected one 4-photon cluster, got {photon_counts[4]}"
    assert photon_counts[2] == 1, f"Expected one 2-photon cluster, got {photon_counts[2]}"
    assert photon_counts[1] == 1, f"Expected one 1-photon cluster, got {photon_counts[1]}"
    assert photon_counts[3] == 0, f"Expected zero 3-photon clusters, got {photon_counts[3]}"

    # Check total photons counted
    assert num_photon_hits == 7, f"Expected 7 photons total, got {num_photon_hits}"

    # Check redistributed charge for a known cluster (430 ADU → 4 photons → 107.5 per photon)
    assert np.isclose(ccd_redistributed[10, 10], 107.5), "Incorrect redistributed charge at pixel (10,10)"
    assert np.isclose(ccd_redistributed[10, 11], 107.5), "Incorrect redistributed charge at pixel (10,11)"
    assert np.isclose(ccd_redistributed[11, 10], 107.5), "Incorrect redistributed charge at pixel (11,10)"

    # Check photon hit map (pixels assigned photons should be marked with '1')
    assert ccd_photon_hits[10, 10] == 1, "Photon hit not correctly marked at (10,10)"
    assert ccd_photon_hits[10, 11] == 1, "Photon hit not correctly marked at (10,11)"
    assert ccd_photon_hits[11, 10] == 1, "Photon hit not correctly marked at (11,10)"

    # Ensure pixels below threshold have no photon hits marked
    assert ccd_photon_hits[40, 40] == 0, "Incorrect photon hit marking at pixel below threshold (40,40)"