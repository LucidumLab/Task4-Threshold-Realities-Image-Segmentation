import numpy as np


def optimal_threshold(image, method_type='global', block_size=11):
    height, width = image.shape
    
    if method_type == 'global':
        corner_pixels = np.concatenate([
            image[:5, :5].flatten(),  
            image[:5, -5:].flatten(),  
            image[-5:, :5].flatten(),  
            image[-5:, -5:].flatten()  
        ])
        T = np.mean(corner_pixels)
        delta = 1
        while delta > 0.1:
            foreground = image[image >= T]
            background = image[image < T]
            m1 = np.mean(foreground) if len(foreground) > 0 else T
            m2 = np.mean(background) if len(background) > 0 else T
            T_new = (m1 + m2) / 2
            delta = abs(T_new - T)
            T = T_new
        thresh = np.zeros_like(image, dtype=np.uint8)
        thresh[image >= T] = 255
        return thresh
    
    elif method_type == 'local':
        if block_size % 2 == 0:
            block_size += 1
        
        thresh = np.zeros_like(image, dtype=np.uint8)
        # Use block_size as the step size for non-overlapping blocks
        step = block_size
        
        for i in range(0, height, step):
            for j in range(0, width, step):
                # Define the block boundaries
                y1 = i
                y2 = min(i + block_size, height)
                x1 = j
                x2 = min(j + block_size, width)
                region = image[y1:y2, x1:x2]
                
                # Iterative thresholding for the block
                T = np.mean(region)
                delta = 1
                while delta > 0.1:
                    foreground = region[region >= T]
                    background = region[region < T]
                    m1 = np.mean(foreground) if len(foreground) > 0 else T
                    m2 = np.mean(background) if len(background) > 0 else T
                    T_new = (m1 + m2) / 2
                    delta = abs(T_new - T)
                    T = T_new
                
                # Apply the threshold to all pixels in the block
                block = image[y1:y2, x1:x2]
                thresh[y1:y2, x1:x2][block >= T] = 255
        
        return thresh
    