# B8 Project – X-ray Single-Photon Energy-Dispersive Spectroscopy Analysis

This repository contains a Python-based framework for analyzing CCD data from X-ray spectroscopy experiments using Bragg diffraction and single-photon counting (SPC). It also includes tools for spectrum reconstruction and synthetic dataset generation for testing and validation.

---

## Project Structure

```
B8_Project_NEW/
├── bragg_engine/                  # Bragg spectroscopy modules
├── spc_engine/                    # SPC and high-ADU photon detection
├── spectral_reconstruction/       # Energy mapping, spectral analysis, visualizations
├── synthetic_engine/              # Synthetic data generation tools
├── synthetic_dataset_example/     # Sample synthetic CCD and energy maps
├── 8/                             # Sample output from image 8 (SPC + high-ADU results)
├── run_bragg_engine.py            # Bragg image analysis
├── run_spc_engine.py              # Single-photon counting and high-ADU clustering
├── run_spectral_reconstruction.py # Spectral reconstruction using photon hits
├── run_synthetic_engine.py        # Generates synthetic image + runs full pipeline
├── run_unit_tests.py              # Unified unit test runner
├── config.py                      # Centralized configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Getting Started

### Clone the Repository

Using SSH (if you have added your SSH key to GitHub) or using HTTPS (recommended for most users):

```bash
git clone git@github.com:HelenaBhattacharya/B8.git
```
---

### Move into project folder
```bash
cd B8
```
Then confirm with:
```bash
ls
```
### Set Up Your Environment
Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

(On Windows, use `venv\Scripts\activate`)

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

Each main component is executed via a `run_*.py` script from the project root.

### Bragg Engine

```bash
python3 run_bragg_engine.py                                   # Uses default sxro6416-r0504.h5 from config.py
python3 run_bragg_engine.py my_file.h5                        # Run with a custom HDF5 input file
python3 run_bragg_engine.py my_new_file.h5 --force-recompute  # Run Bragg Engine to regenerate optimized_params.npy and quadratic_params.npy
```

> ⚠️ Rerunning the CMA-ES optimisation can take **several minutes**.

---

### SPC Engine

```bash
python3 run_spc_engine.py                                 # Run SPC over all 20 images
python3 run_spc_engine.py --single_image 8                # Run SPC on image index 8 only
python3 run_spc_engine.py my_file.h5 --single_image 3
```

> ⚠️ Running SPC over all 20 images may take **several hours**, depending on your machine.  To skip this step and save time, use the precomputed outputs in folders 0/ to 19/, already included in the repository.

---

### Spectral Reconstruction

```bash
python3 run_spectral_reconstruction.py                     # Run over all available processed images
python3 run_spectral_reconstruction.py --image_index 7     # Run for a single image
python3 run_spectral_reconstruction.py --images 1 2 4 6 8  # Run for several images
python3 run_spectral_reconstruction.py my_file.h5
```

---

### Synthetic Engine

```bash
python3 run_synthetic_engine.py
```

This generates a synthetic CCD image using realistic photon hits and immediately runs the full Bragg + SPC + spectral reconstruction pipeline.

---

### Run Unit Tests

Unit tests validating all components (Bragg, SPC, spectral reconstruction) can be run using:

```bash
python3 run_unit_tests.py
```

---

## Notes on Sample Output

Folders named `0/` through `19/` are included in the repository to serve as **precomputed reference outputs**. Each folder corresponds to a specific CCD image index and contains the full output of the SPC + high-ADU clustering pipeline, including:
- Detected single-photon clusters (cluster_pixel_map_*.npy)
- ADU-weighted CCD maps (adu_weighted_ccd_final_*.npy)
- High-ADU cluster data (high_ADU_clusters_*.pkl)
- final redistributed charge maps (ccd_redistributed_*.npy)

This allows users to explore and analyse the **spectral reconstruction pipeline without reprocessing** the raw CCD images.

To regenerate all outputs locally, simply run:
```bash
python3 run_spc_engine.py
```

---

## Author

**Helena Bhattacharya**
