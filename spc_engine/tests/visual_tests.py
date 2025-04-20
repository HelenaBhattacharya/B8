import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import generic_filter
from spc_engine.spc import detect_photon_events
from spc_engine.high_ADU import redistribute_charge
from bragg_engine.load import get_ccd_image
from config import HDF5_FILE, CCD_SHAPE
# import matplotlib
# matplotlib.use('TkAgg')

# --- Real Image Processing ---
def detect_single_photon_hits(image, photon_threshold=130):
    """
    Detects isolated single-pixel photon hits above a given threshold.
    """
    def photon_filter(values):
        center = values[4]
        surrounding = np.array([values[i] for i in range(9) if i != 4])
        return 1 if (center > photon_threshold and np.all(surrounding < photon_threshold)) else 0

    binary_image = generic_filter(image, photon_filter, size=(3, 3), mode='constant', cval=0)
    y_coords, x_coords = np.where(binary_image == 1)
    photon_count = len(x_coords)

    return binary_image, x_coords, y_coords, photon_count


# --- Synthetic CCD Generation ---
def generate_simple_synthetic_image(ccd_size=(200, 200)):
    """
        Generates a basic synthetic CCD image with clean SPC cluster shapes.
        """
    synthetic_ccd_image = np.zeros(ccd_size, dtype=np.float32)

    cluster_shapes = {
        1: [np.array([[1]])],
        2: [np.array([[1, 1]]), np.array([[1], [1]])],
        3: [
            np.array([[1, 1, 0], [0, 1, 0]]), np.array([[1, 0], [1, 1]]),
            np.array([[1, 1], [1, 0]]), np.array([[1, 1], [0, 1]])
        ],
        4: [np.array([[1, 1], [1, 1]])]
    }

    num_clusters = {1: 20, 2: 30, 3: 40, 4: 50}

    def distribute_adu(num_pixels, min_adu=100, max_adu=200):
        total_adu = random.randint(min_adu, max_adu)
        adu_values = np.random.uniform(15, total_adu / num_pixels, num_pixels)
        adu_values *= total_adu / np.sum(adu_values)
        return adu_values

    for size, count in num_clusters.items():
        for _ in range(count):
            shape = random.choice(cluster_shapes[size])
            rotated_shape = np.rot90(shape, random.randint(0, 3))
            h, w = rotated_shape.shape

            x, y = random.randint(0, ccd_size[0] - h), random.randint(0, ccd_size[1] - w)
            adu_values = distribute_adu(np.sum(rotated_shape))

            idx = 0
            for i in range(h):
                for j in range(w):
                    if rotated_shape[i, j]:
                        synthetic_ccd_image[x + i, y + j] = adu_values[idx]
                        idx += 1

    return synthetic_ccd_image

def generate_complex_synthetic_image(ccd_size=(200, 200)):
    """
    Generates a synthetic CCD image with overlapping and composite clusters.
    """
    complex_image = np.zeros(ccd_size, dtype=np.float32)

    # Defined shapes and composite types
    single_cluster_shapes = {
        1: [np.array([[1]])],
        2: [np.array([[1, 1]]), np.array([[1], [1]])],
        3: [
            np.array([[1, 1, 0], [0, 1, 0]]), np.array([[1, 0], [1, 1]]),
            np.array([[1, 1], [1, 0]]), np.array([[1, 1], [0, 1]])
        ],
        4: [np.array([[1, 1], [1, 1]])]
    }

    composite_cluster_types = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

    num_single_clusters = {1: 80, 2: 70, 3: 60, 4: 50}
    num_composite_clusters = {(1, 2): 20, (1, 3): 20, (1, 4): 20,
                              (2, 3): 15, (2, 4): 15, (3, 4): 10}

    def distribute_adu(num_pixels, min_adu, max_adu):
        total_adu = random.randint(min_adu, max_adu)
        adu_values = np.random.uniform(15, total_adu / num_pixels, num_pixels)
        adu_values *= total_adu / np.sum(adu_values)
        return adu_values

    def transform_shape(shape):
        shape = np.rot90(shape, random.randint(0, 3))
        if random.choice([True, False]):
            shape = np.fliplr(shape)
        if random.choice([True, False]):
            shape = np.flipud(shape)
        return shape

    # Insert single clusters
    for size, count in num_single_clusters.items():
        for _ in range(count):
            shape = transform_shape(random.choice(single_cluster_shapes[size]))
            h, w = shape.shape
            x, y = random.randint(0, ccd_size[0]-h), random.randint(0, ccd_size[1]-w)
            adu_values = distribute_adu(np.count_nonzero(shape), 100, 200)

            idx = 0
            for i in range(h):
                for j in range(w):
                    if shape[i, j]:
                        complex_image[x+i, y+j] = adu_values[idx]
                        idx += 1

    # Insert composite (High-ADU) clusters
    for (size1, size2), count in num_composite_clusters.items():
        for _ in range(count):
            shape1 = transform_shape(random.choice(single_cluster_shapes[size1]))
            shape2 = transform_shape(random.choice(single_cluster_shapes[size2]))

            max_h, max_w = max(shape1.shape[0], shape2.shape[0]), max(shape1.shape[1], shape2.shape[1])
            composite_shape = np.zeros((max_h+1, max_w+1))

            composite_shape[:shape1.shape[0], :shape1.shape[1]] += shape1
            offset_x, offset_y = random.randint(0, 1), random.randint(0, 1)
            composite_shape[offset_x:offset_x+shape2.shape[0], offset_y:offset_y+shape2.shape[1]] += shape2

            composite_shape[composite_shape > 0] = 1

            h, w = composite_shape.shape
            x, y = random.randint(0, ccd_size[0]-h), random.randint(0, ccd_size[1]-w)
            adu_values = distribute_adu(np.count_nonzero(composite_shape), 300, 500)  # High-ADU threshold

            idx = 0
            for i in range(h):
                for j in range(w):
                    if composite_shape[i, j]:
                        complex_image[x+i, y+j] = adu_values[idx]
                        idx += 1

    return complex_image

# --- Visual Test Using Real Data ---
def visual_test_SPSP_real_data(HDF5_FILE_PATH=HDF5_FILE, image_index=8, photon_threshold=110):
    """
    Visual test for detecting SPSP hits in real CCD image data.
    """
    ccd_image = get_ccd_image(str(HDF5_FILE_PATH), image_index)
    binary_ccd, x_hits, y_hits, num_photon_hits = detect_single_photon_hits(ccd_image, photon_threshold)

    print(f"\n--- SPSP Real Data Visual Test ---")
    print(f"Total detected single-photon hits: {num_photon_hits}")

    plt.figure(figsize=(8, 6))
    plt.imshow(binary_ccd, cmap='hot', origin='upper', vmin=0, vmax=1)
    plt.scatter(x_hits, y_hits, color='cyan', s=5, marker='o', label='Single-Photon Hits')
    plt.colorbar(label='Single-Photon Detection (1 = Hit, 0 = No Hit)')
    plt.title('Single-Photon, Single-Pixel Hits (Real Data)')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.legend()
    plt.show()


# --- Visual Test Using SPC and High-ADU ---
def visual_test_synthetic_processing(image_type='simple'):
    """
    Runs full SPC + high-ADU detection on synthetic CCD image and plots results.
    """
    if image_type == 'simple':
        ccd_image = generate_simple_synthetic_image()
    elif image_type == 'complex':
        ccd_image = generate_complex_synthetic_image()
    else:
        raise ValueError("Invalid image type. Choose 'simple' or 'complex'.")

    adu_weighted_ccd, _, photon_events, high_ADU_clusters, _, _ = detect_photon_events(
        ccd_image, sigma_N=9.5, ADU_sp=120
    )

    if image_type == 'complex':
        _, redistributed, photon_hits_map, _, photon_counts = redistribute_charge(
            high_ADU_clusters, ccd_size=ccd_image.shape
        )
    else:
        photon_hits_map = np.zeros_like(ccd_image)
        photon_counts = {}

    # Print cluster summary
    print(f"\n{'='*10} {image_type.capitalize()} Synthetic Test {'='*10}")
    for size in range(1, 5):
        print(f"SPC {size}-pixel hits: {len(photon_events[size])}")
    if image_type == 'complex':
        print("\nHigh-ADU clusters:")
        for photons, count in sorted(photon_counts.items()):
            print(f"{photons}-photon clusters: {count}")

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(ccd_image, cmap='hot')
    colors = {1: 'cyan', 2: 'blue', 3: 'green', 4: 'magenta'}

    for size, hits in photon_events.items():
        if hits:
            x, y, _ = zip(*hits)
            plt.scatter(y, x, color=colors[size], s=1, label=f'SPC {size}-pixel')

    if image_type == 'complex':
        y_high, x_high = np.where(photon_hits_map)
        plt.scatter(x_high, y_high, color='white', s=1, label='High-ADU Hits')

    plt.title(f'{image_type.capitalize()} Synthetic CCD')
    plt.legend()
    plt.xlabel('Pixel X'); plt.ylabel('Pixel Y')
    plt.colorbar(label='ADU')
    plt.tight_layout()
    plt.show()

# --- Manual Runner ---
if __name__ == "__main__":
    visual_test_synthetic_processing('simple')
    visual_test_synthetic_processing('complex')
    visual_test_SPSP_real_data()
