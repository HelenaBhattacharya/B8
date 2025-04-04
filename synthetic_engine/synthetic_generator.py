import numpy as np
import random
from bragg_engine.mapping import compute_energy_map


# === Cluster shape templates ===
SPC_SHAPES = {
    1: [np.array([[1]])],
    2: [np.array([[1, 1]]), np.array([[1], [1]])],
    3: [
        np.array([[1, 1, 0], [0, 1, 0]]),
        np.array([[1, 0], [1, 1]]),
        np.array([[1, 1], [1, 0]]),
        np.array([[1, 1], [0, 1]])
    ],
    4: [np.array([[1, 1], [1, 1]])]
}


# === Modular Functions ===

def build_energy_map(optimized_params):
    return compute_energy_map(optimized_params)[0]


def synthetic_spectrum(E_map):
    """Construct synthetic spectrum (lines + flat + exponential tail)."""
    spectrum = np.zeros_like(E_map)

    # Spectral lines (Gaussian-shaped)
    lines = [
        {'E': 1188.0, 'sigma': 2.0, 'intensity': 1.0},
        {'E': 1218.5, 'sigma': 2.0, 'intensity': 0.85},
    ]
    for line in lines:
        spectrum += line['intensity'] * np.exp(-((E_map - line['E']) ** 2) / (2 * line['sigma'] ** 2))

    # Flat background between 1000–1200 eV
    flat_mask = (E_map >= 1000) & (E_map <= 1185)
    spectrum += 0.2 * flat_mask

    # Bremsstrahlung-like exponential tail >1200 eV
    brem_mask = E_map > 1185
    spectrum += 0.5 * np.exp(-0.005 * (E_map - 1185)) * brem_mask

    return spectrum / np.max(spectrum)


def get_cluster_parameters(photon_count):
    """Return (mean ADU, sigma) for a given cluster size."""
    # Gaussian peaks get broader and dimmer with n
    mean_table = {1: 160, 2: 320, 3: 480, 4: 640}
    sigma_table = {1: 15, 2: 25, 3: 35, 4: 45}
    return mean_table[photon_count], sigma_table[photon_count]


def distribute_adu_gaussian(shape_mask, photon_count):
    """Assign total ADU using a Gaussian distribution, then spread across shape."""
    mean, sigma = get_cluster_parameters(photon_count)
    total_adu = np.random.normal(loc=mean, scale=sigma)
    total_adu = np.clip(total_adu, 50, 800)

    n_pixels = np.count_nonzero(shape_mask)
    weights = np.sort(np.random.exponential(scale=1.0, size=n_pixels))[::-1]
    weights /= np.sum(weights)
    adus = total_adu * weights

    return adus


def place_cluster(ccd, label_map, x0, y0, cluster_type='single', max_photon_count=4):
    """Place a single or composite cluster at (x0, y0) on CCD and update maps."""
    ny, nx = ccd.shape

    def transform(shape):
        shape = np.rot90(shape, random.randint(0, 3))
        if random.choice([True, False]):
            shape = np.flipud(shape)
        if random.choice([True, False]):
            shape = np.fliplr(shape)
        return shape

    # === Select cluster type ===
    if cluster_type == 'single':
        photons = np.random.choice([1, 2, 3, 4], p=[0.7, 0.2, 0.08, 0.02])
        shape = transform(random.choice(SPC_SHAPES[photons]))
        label_value = photons

    elif cluster_type == 'composite':
        p1, p2 = sorted(random.sample([1, 2, 3, 4], 2))
        shape1 = transform(random.choice(SPC_SHAPES[p1]))
        shape2 = transform(random.choice(SPC_SHAPES[p2]))

        h = max(shape1.shape[0], shape2.shape[0]) + 1
        w = max(shape1.shape[1], shape2.shape[1]) + 1
        composite = np.zeros((h, w), dtype=int)

        composite[:shape1.shape[0], :shape1.shape[1]] += shape1
        dy, dx = random.randint(0, 1), random.randint(0, 1)
        composite[dy:dy+shape2.shape[0], dx:dx+shape2.shape[1]] += shape2
        composite[composite > 0] = 1
        shape = composite
        photons = p1 + p2
        label_value = photons

    else:
        raise ValueError("Invalid cluster_type")

    h, w = shape.shape
    if x0 + w >= nx or y0 + h >= ny:
        return  # Skip if shape would go out of bounds

    adus = distribute_adu_gaussian(shape, min(photons, max_photon_count))

    # Place ADUs into CCD + label map
    idx = 0
    for i in range(h):
        for j in range(w):
            if shape[i, j]:
                ccd[y0 + i, x0 + j] += adus[idx]
                label_map[y0 + i, x0 + j] += label_value
                idx += 1
