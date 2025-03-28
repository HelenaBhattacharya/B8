# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Central data file and CCD shape
HDF5_FILE = PROJECT_ROOT / "sxro6416-r0504.h5"
CCD_SHAPE = (2048, 2048)
MAX_IMAGES = 20

# Preprocessing parameters
THRESHOLD_PERCENTILE = 90
HIGH_ADU_THRESHOLD = 3000

# Geometry and detector setup
PIXEL_SIZE = 13.5e-6  # meters
LATTICE_SPACING = 7.98e-10  # meters
ALPHA_Z = 0  # (if needed in future)

# Bragg optimization constants
PARAM_BOUNDS = [
    (0.08, 0.09),  # D (m)
    (37, 38),  # θ̄_B (degrees)
    (-10, 10),  # α_x (degrees)
    (30, 70),  # α_y (degrees)
]
INITIAL_PARAMS = [0.083, 37.2, -1.89, 55.8]
ENERGY_LEVELS = [1188, 1218.5]
CURVE_CENTER_1188_INIT = 1424
CURVE_CENTER_1218_INIT = 1284

SPOT_AREA_M2 = 28.5e-12

CCD_CENTER_X = CCD_SHAPE[1] // 2  # 1024
CCD_CENTER_Y = CCD_SHAPE[0] // 2  # 1024

# Pedestal processing
PEDESTAL_BIN_WIDTH = 2          # ADU bin width for histogram
PEDESTAL_FIT_RANGE = (0, 100)   # ADU range for fitting
PEDESTAL_SIGMA_INIT = 10        # Initial guess for sigma

# SPC parameters
SPC_SIGMA_N = 9.5
SPC_ADU_SINGLE_PHOTON = 120
SPC_FANO_FACTOR = 0.12
SPC_CLUSTER_SUM_MAX = 200  # Optional — used for cluster filtering
SPC_HIGH_ADU_PIXEL_THRESHOLD = 50

# High ADU clustering and redistribution
HIGH_ADU_PIXEL_THRESHOLD = 20
HIGH_ADU_CLUSTER_THRESHOLDS = {
    1: 200,
    2: 300,
    3: 400
}  # Thresholds for determining number of photons per cluster

# Spectral reconstruction settings
OPTIMIZED_PARAMS_FILE = PROJECT_ROOT / "optimized_params.npy"
QUADRATIC_PARAMS_FILE = PROJECT_ROOT / "quadratic_params.npy"
ENERGY_MIN = 1100
ENERGY_MAX = 1600
BIN_WIDTH = 1
