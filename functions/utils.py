import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from scipy.signal import convolve2d

def compute_histogram(data, levels=256):
    hist, _ = np.histogram(data.flatten(), bins=levels, range=(0, levels), density=True)
    return hist


def compute_local_means(image, block_size):
    if block_size % 2 == 0:
        block_size += 1
    kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size * block_size)
    local_means = convolve2d(image, kernel, mode='same', boundary='symmetric')
    return local_means