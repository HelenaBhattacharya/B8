import numpy as np
import json
from scipy.ndimage import gaussian_filter


class CCDDetector:
    def __init__(self, param_file):
        # Load from JSON config
        with open(param_file, 'r') as file:
            params = json.load(file)

        self.ny, self.nx = params["array_size"]
        self.pixel_size_m = params["pixel_size_microns"] * 1e-6
        self.dark_current_e_per_px_s = params["dark_current_e_per_px_s"]
        self.readout_noise_e_rms = params["readout_noise_e_rms"]
        self.adc_bits = params["adc_bit_depth"]
        self.adc_gain = params["adc_gain_e_per_adu"]
        self.full_well_capacity_e = params["full_well_capacity_e"]
        self.charge_spread_sigma = params["charge_spread_sigma_pixels"]

        # Derived parameters
        self.max_adu = 2**self.adc_bits - 1
        self.pedestal_mean = 60     # in ADU
        self.pedestal_sigma = 10    # in ADU

    def generate_dark_current(self, exposure_s):
        """Simulate dark current electrons for each pixel (Poisson)"""
        return np.random.poisson(
            self.dark_current_e_per_px_s * exposure_s,
            size=(self.ny, self.nx)
        )

    def add_photon_hits(self, electron_map, photon_hits):
        """Add electrons to frame from photon hits (before spreading)"""
        for y, x, electrons in photon_hits:
            if 0 <= y < self.ny and 0 <= x < self.nx:
                electron_map[y, x] += electrons
        return electron_map

    def apply_charge_spread(self, electron_map):
        """Apply Gaussian PSF to simulate charge spread"""
        return gaussian_filter(electron_map, sigma=self.charge_spread_sigma)

    def apply_readout_noise(self, electron_map):
        """Add Gaussian readout noise (in electrons)"""
        noise = np.random.normal(
            loc=0,
            scale=self.readout_noise_e_rms,
            size=(self.ny, self.nx)
        )
        return electron_map + noise

    def convert_to_adu(self, electron_map):
        """Convert electrons to ADU using ADC gain, clip to bit depth"""
        adu_map = electron_map / self.adc_gain
        adu_map = np.clip(adu_map, 0, self.max_adu)
        return adu_map

    def add_pedestal(self, adu_map):
        """Add Gaussian pedestal (simulated background bias)"""
        pedestal = np.random.normal(
            loc=self.pedestal_mean,
            scale=self.pedestal_sigma,
            size=(self.ny, self.nx)
        )
        final = adu_map + pedestal
        return np.clip(final, 0, self.max_adu).astype(np.uint16)

    def acquire_frame(self, exposure_s=1.0, photon_hits=None):
        """Full CCD simulation from photon hits to ADU image"""
        if photon_hits is None:
            photon_hits = []

        frame_electrons = self.generate_dark_current(exposure_s)
        frame_electrons = self.add_photon_hits(frame_electrons, photon_hits)
        frame_electrons = self.apply_charge_spread(frame_electrons)
        frame_electrons = np.clip(frame_electrons, 0, self.full_well_capacity_e)
        frame_electrons = self.apply_readout_noise(frame_electrons)
        frame_adu = self.convert_to_adu(frame_electrons)
        return self.add_pedestal(frame_adu)