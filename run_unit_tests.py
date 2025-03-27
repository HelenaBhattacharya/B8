import pytest
import os
import bragg_engine.tests.visual_tests as bragg_visuals
import spc_engine.tests.visual_tests as spc_visuals
import spectral_reconstruction.tests.visual_tests as spectral_visuals
from config import HDF5_FILE
# Ensure working directory is project root
os.chdir(os.path.dirname(__file__))

def run_bragg_unit_tests():
    print("\n=== Running Bragg Unit Tests ===")
    result = pytest.main([
        "bragg_engine/tests",
        "-m", "not slow and not visual",
        "--tb=short"
    ])
    if result != 0:
        print("❌ Bragg unit tests failed.")
    else:
        print("✅ Bragg unit tests passed.")

def run_spc_unit_tests():
    print("\n=== Running SPC Unit Tests ===")
    result = pytest.main([
        "spc_engine/tests",
        "-m", "not slow and not visual",
        "--tb=short"
    ])
    if result != 0:
        print("❌ SPC unit tests failed.")
    else:
        print("✅ SPC unit tests passed.")

def run_spectral_unit_tests():
    print("\n=== Running Spectral Unit Tests ===")
    result = pytest.main([
        "spectral_reconstruction/tests",
        "-m", "not slow and not visual",
        "--tb=short"
    ])
    if result != 0:
        print("❌ Spectral unit tests failed.")
    else:
        print("✅ Spectral unit tests passed.")

def run_bragg_visual_tests():
    print("\n=== Running Bragg Visual Tests ===")
    try:
        bragg_visuals.visual_solid_angle_test()
        bragg_visuals.visual_energy_dispersion_test()
        bragg_visuals.visual_optimized_vs_experimental_test()
        print("✅ Bragg visual tests completed.\n")
    except Exception as e:
        print(f"❌ Bragg visual tests failed: {e}")

def run_spc_visual_tests():
    print("\n=== Running SPC Visual Tests ===")
    try:
        spc_visuals.visual_test_synthetic_processing("simple")
        spc_visuals.visual_test_synthetic_processing("complex")
        spc_visuals.visual_test_SPSP_real_data()
        print("✅ SPC visual tests completed.\n")
    except Exception as e:
        print(f"❌ SPC visual tests failed: {e}")

def run_spectral_visual_tests():
    print("\n=== Running Spectral Visual Tests ===")
    try:
        spectral_visuals.visual_test_photon_energy_histogram()
        spectral_visuals.visual_test_overlap_resolution()
        spectral_visuals.visual_test_simple_0th_order_spectrum()
        import numpy as np
        from config import OPTIMIZED_PARAMS_FILE
        optimized_params = np.load(str(OPTIMIZED_PARAMS_FILE), allow_pickle=True)
        spectral_visuals.visual_test_simple_1st_order_spectrum(file_name=str(HDF5_FILE),optimized_params=optimized_params)
        spectral_visuals.visual_test_2nd_order_spectrum(file_name=str(HDF5_FILE),optimized_params=optimized_params)
        print("✅ Spectral visual tests completed.\n")
    except Exception as e:
        print(f"❌ Spectral visual tests failed: {e}")

def run_all_tests():
    run_bragg_unit_tests()
    run_spc_unit_tests()
    run_spectral_unit_tests()
    run_bragg_visual_tests()
    run_spc_visual_tests()
    run_spectral_visual_tests()
    print("\n=== All Bragg + SPC + Spectral Tests Completed ===")


if __name__ == "__main__":
    run_all_tests()