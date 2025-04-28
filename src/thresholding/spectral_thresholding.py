import numpy as np

from .utils import  compute_local_means


def spectral_threshold(image, method_type='global', block_size=11, num_levels=3,offset_range=10):
    height, width = image.shape
    
    if num_levels < 2:
        raise ValueError("Number of levels must be at least 2 (for 1 threshold).")
    
    # Calculate intensity levels (evenly spaced from 0 to 255)
    levels = np.linspace(0, 255, num_levels).astype(np.uint8)
    
    if method_type == 'global':
        pixels = image.flatten().reshape(-1, 1)
        pixels = pixels.astype(np.float32)
        # Initialize clusters at evenly spaced intervals
        centers = np.linspace(0, 255, num_levels, dtype=np.float32).reshape(-1, 1)
        labels = np.zeros(len(pixels), dtype=np.int32)
        for _ in range(10):
            distances = np.abs(pixels - centers[:, np.newaxis])
            labels = np.argmin(distances, axis=0)
            for k in range(num_levels):
                if np.sum(labels == k) > 0:
                    centers[k] = np.mean(pixels[labels == k])
        centers = np.sort(centers.flatten())
        result = np.zeros_like(labels, dtype=np.uint8)
        for k in range(num_levels):
            result[labels == k] = levels[k]
        return result.reshape(height, width)
    
    elif method_type == 'local':
        if block_size % 2 == 0:
            block_size += 1
        half_block = block_size // 2
        thresh = np.zeros_like(image, dtype=np.uint8)

        # Use convolution to compute local means
        local_means = compute_local_means(image, block_size)

        num_thresholds = num_levels - 1
        offsets = np.linspace(-offset_range / 2, offset_range / 2, num_thresholds)

        for i in range(height):
            for j in range(width):
                mean = local_means[i, j]
                thresholds = mean + offsets
                pixel = image[i, j]
                level_idx = 0
                for t in range(num_thresholds):
                    if pixel > thresholds[t]:
                        level_idx = t + 1
                    else:
                        break
                thresh[i, j] = levels[level_idx]
        return thresh
