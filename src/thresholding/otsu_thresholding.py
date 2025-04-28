import numpy as np
from .utils import compute_histogram, compute_local_means

def otsu_threshold(image, method_type='global', block_size=11):
    height, width = image.shape
    
    if method_type == 'global':
        hist = compute_histogram(image)
        total_pixels = height * width
        hist = hist * total_pixels
        sum_all = np.sum(np.arange(256) * hist)
        sum_b = 0
        w_b = 0
        max_variance = 0
        T = 0
        
        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total_pixels - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_all - sum_b) / w_f
            variance = w_b * w_f * (m_b - m_f) ** 2
            if variance > max_variance:
                max_variance = variance
                T = t
        
        thresh = np.zeros_like(image, dtype=np.uint8)
        thresh[image >= T] = 255
        return thresh
    
    elif method_type == 'local':
        if block_size % 2 == 0:
            block_size += 1
        
        # Compute local means using convolution
        local_means = compute_local_means(image, block_size)
        
        T = local_means
        
        # Apply threshold vectorized
        thresh = np.zeros_like(image, dtype=np.uint8)
        thresh[image >= T] = 255
        
        return thresh
    