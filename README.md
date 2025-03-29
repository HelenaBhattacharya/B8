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

Using SSH (if you have added your SSH key to GitHub):

```bash
git clone git@github.com:HelenaBhattacharya/B8.git
cd B8_Project_NEW
```

Or using HTTPS (recommended for most users):

```bash
git clone https://github.com/HelenaBhattacharya/B8.git
cd B8_Project_NEW
```

---

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
python3 run_bragg_engine.py               # Uses default sxro6416-r0504.h5 from config.py
python3 run_bragg_engine.py my_file.h5    # Run with a custom HDF5 input file
```

---

### SPC Engine

```bash
python3 run_spc_engine.py                         # Run SPC over all 20 images
python3 run_spc_engine.py --single_image 8        # Run SPC on image index 8 only
python3 run_spc_engine.py my_file.h5 --single_image 3
```

> ⚠️ Running SPC over all 20 images may take **several hours**, depending on your machine.

---

### Spectral Reconstruction

```bash
python3 run_spectral_reconstruction.py                 # Run over all available processed images
python3 run_spectral_reconstruction.py --image_index 7 # Run for a single image
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
python run_unit_tests.py
```

---

## Notes on Sample Output

A folder named `8/` is included in the repository to serve as a **precomputed sample**. It contains the full output of SPC + high-ADU clustering for image index 8, including:
- Detected single-photon clusters
- ADU-weighted CCD images
- Photon hit maps
- Final redistributed charge images

This allows users to **run the spectral reconstruction pipeline immediately**, without first processing all 20 CCD images.

To regenerate all image outputs locally, simply run:

```bash
python run_spc_engine.py
```

---

## Author

**Helena Bhattacharya**
