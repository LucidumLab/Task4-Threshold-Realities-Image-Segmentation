import numpy as np
from scipy.spatial import cKDTree
import random

def mean_shift_segmentation_without_boundries(image_array, threshold=30, convergence_threshold=1.0, max_iterations=1000):
    """
    Segments an image based on color and spatial proximity using mean shift clustering.
    
    Parameters:
        image_array (numpy.ndarray): Input image as a NumPy array (BGR format).
        threshold (float): Distance threshold for clustering.
        convergence_threshold (float): Threshold for cluster center convergence.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.
    
    Returns:
        segmented_image (numpy.ndarray): Segmented image.
    """
    row, col, _ = image_array.shape

    
    i, j = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
    rgb = image_array.reshape(-1, 3)
    ij = np.stack([i.ravel(), j.ravel()], axis=1)
    D = np.hstack([rgb, ij]).astype(np.float32)

    
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

        if mean_shift_distance < convergence_threshold:
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






def create_feature_space(image):
    """Converts the image into a 5D feature space (R, G, B, i, j) using vectorized operations."""
    row, col, _ = image.shape
   
    i, j = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
   
    rgb = image.reshape(-1, 3)
    ij = np.stack([i.ravel(), j.ravel()], axis=1)
    D = np.hstack([rgb, ij]).astype(np.float32)
    return D, row, col

def mean_shift_segmentation_with_extra(D, row, col, threshold=30, convergence_threshold=0.01, max_iterations=1000):
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

        if mean_shift_distance < convergence_threshold:
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
    segmented_image = mean_shift_segmentation_with_extra(feature_space, row, col, threshold=150, convergence_threshold=1, max_iterations=1000)
    return segmented_image

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "butterfly.png")
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Image could not be loaded")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(image_rgb)
        plt.axis('off')
        
        start_time = time.time()
        segmented_image1 = mean_shift_segmentation_without_boundries(image, threshold=30)
        segmented_image1_rgb = cv2.cvtColor(segmented_image1, cv2.COLOR_BGR2RGB)
        end_time = time.time()
        print(f"Method 1 execution time: {end_time - start_time:.2f} seconds")
        
        start_time = time.time()
        segmented_image2 = mean_shift_from_array(image)
        segmented_image2_rgb = cv2.cvtColor(segmented_image2, cv2.COLOR_BGR2RGB)
        end_time = time.time()
        print(f"Method 2 execution time: {end_time - start_time:.2f} seconds")
        
        plt.subplot(132)
        plt.title("Method 1: Without Boundaries")
        plt.imshow(segmented_image1_rgb)
        plt.axis('off')
        
        plt.subplot(133)
        plt.title("Method 2: With Extra")
        plt.imshow(segmented_image2_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        cv2.imwrite("segmented_method1.jpg", segmented_image1)
        cv2.imwrite("segmented_method2.jpg", segmented_image2)
        print("Segmented images saved as 'segmented_method1.jpg' and 'segmented_method2.jpg'")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()