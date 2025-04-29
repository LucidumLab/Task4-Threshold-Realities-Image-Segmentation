import torch
import torch.nn.functional as F
import torchvision.transforms as transforms  
def cluster_image_kmeans(image, k=5, max_iter=100, device='cuda'):
    """
    Perform K-means clustering on an image using PyTorch (GPU accelerated).

    Parameters:
    - image: input RGB image (torch.Tensor) with shape (H, W, 3)
    - k: number of clusters
    - max_iter: maximum number of KMeans iterations
    - device: device to perform computation on ('cuda' or 'cpu')

    Returns:
    - clustered_image: image recolored by cluster centroids (torch.Tensor)
    """
    # Move image to device and reshape
    pixels = image.to(device).view(-1, 3).float()
    
    # Initialize centroids using random pixels
    centroids = pixels[torch.randperm(pixels.shape[0])[:k]].clone()
    
    for _ in range(max_iter):
        # Compute distances between pixels and centroids
        distances = torch.cdist(pixels, centroids)
        
        # Assign pixels to nearest centroid
        assignment = torch.argmin(distances, dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = (assignment == i)
            if mask.any():
                new_centroids[i] = pixels[mask].mean(dim=0)
        
        # Check for convergence
        if torch.allclose(centroids, new_centroids, rtol=1e-4):
            break
            
        centroids = new_centroids
    
    # Reconstruct the clustered image
    clustered_pixels = centroids[assignment]
    clustered_image = clustered_pixels.view(image.shape)
    
    return clustered_image

# Example usage:
if __name__ == "__main__":
    # Load an image and convert to tensor
    from PIL import Image
    import numpy as np
    from torchvision import transforms
    
    # Load image
    img = Image.open("your_image.jpg")
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)  # Convert to HxWxC format
    
    # Perform clustering on GPU
    clustered_img = cluster_image_kmeans(img_tensor, k=5)
    
    # Convert back to PIL image for display/saving
    clustered_img_np = clustered_img.cpu().numpy()
    clustered_img_pil = Image.fromarray((clustered_img_np * 255).astype('uint8'))
    clustered_img_pil.save("clustered_image.jpg")