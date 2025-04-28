import numpy as np
import cv2
from scipy.spatial import cKDTree
import random
import matplotlib.pyplot as plt


def create_feature_space(image):
    """Converts the image into a 5D feature space (R, G, B, i, j) using vectorized operations."""
    row, col, _ = image.shape
   
    i, j = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
   
    rgb = image.reshape(-1, 3)
    ij = np.stack([i.ravel(), j.ravel()], axis=1)
    D = np.hstack([rgb, ij]).astype(np.float32)
    return D, row, col

def mean_shift_segmentation(D, row, col, threshold=30, convergence_thresh=0.01, max_iterations=1000):
    """Performs mean shift clustering with KD-tree optimization and returns the segmented image."""
    segmented_image = np.zeros((row, col, 3), dtype=np.uint8)
    current_mean_random = True
    current_mean = np.zeros(5, dtype=np.float32)
    iteration = 0

   
    tree = cKDTree(D)

    while len(D) > 0 and iteration < max_iterations:
        if current_mean_random:
            random_idx = random.randint(0, len(D) - 1)
            current_mean = D[random_idx].copy()

       
        below_threshold_indices = tree.query_ball_point(current_mean, r=threshold)

        if not below_threshold_indices:
            current_mean_random = True
            iteration += 1
            continue

       
        cluster_points = D[below_threshold_indices]
        new_mean = np.mean(cluster_points, axis=0)

       
        mean_shift_distance = np.linalg.norm(new_mean - current_mean)

        if mean_shift_distance < convergence_thresh:
            cluster_color = new_mean[:3].astype(np.uint8)
            for idx in below_threshold_indices:
                r, c = int(D[idx, 3]), int(D[idx, 4])
                segmented_image[r, c] = cluster_color

           
            D = np.delete(D, below_threshold_indices, axis=0)
            tree = cKDTree(D) 
            current_mean_random = True
        else:
            current_mean = new_mean
            current_mean_random = False

        iteration += 1

    return segmented_image

def mean_shift_from_array(image_array):
    """Main function: takes a NumPy image array and returns segmented image."""
    feature_space, row, col = create_feature_space(image_array)
    segmented_image = mean_shift_segmentation(feature_space, row, col, threshold=150, convergence_thresh=1, max_iterations=1000)
    return segmented_image


def display_image(window_name, image):
    """Displays an image using Matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    plt.figure(window_name)
    plt.imshow(image_rgb)
    plt.title(window_name)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image = cv2.imread("butterfly.png")
    if image is None:
        raise ValueError("Could not read the image. Check the path.")
    
    segmented = mean_shift_from_array(image)
    display_image("Segmented Image", segmented)