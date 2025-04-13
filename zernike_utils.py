#zernike_utils.py
import os
import cv2
import numpy as np
import mahotas
import matplotlib.pyplot as plt

def generate_zernike_map(image_path, save_dir, radius=128, degree=12):
    """
    Generate Zernike polynomial-based topography map from corneal image.
    
    Args:
        image_path: Path to input image
        save_dir: Directory to save output map
        radius: Radius for Zernike analysis
        degree: Degree of Zernike polynomials
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {image_path}")
        return
    
    # Resize to 256x256
    img = cv2.resize(img, (256, 256))
    
    # Create circular mask to focus on corneal area
    mask = np.zeros_like(img, dtype=np.uint8)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask to image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Get Zernike moments
    moments = mahotas.features.zernike_moments(masked_img, radius=radius, degree=degree)
    
    # Create a more realistic topography map using Zernike coefficients
    # This is a simplified approach, ideally you would reconstruct the surface using proper
    # Zernike polynomial reconstruction
    z_map = np.zeros((256, 256), dtype=np.float32)
    
    # Create coordinate grid
    y, x = np.indices((256, 256))
    y = y - 128
    x = x - 128
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Only reconstruct within the corneal area
    idx = r <= radius
    
    # Simplified reconstruction (would be better with proper Zernike polynomials)
    # This is just an approximation for visualization
    for i, coef in enumerate(moments[:20]):  # Use first 20 moments
        weight = coef * 10  # Scale factor
        if i % 2 == 0:  # Even indices
            z_map[idx] += weight * np.cos(i * theta[idx] / 2) * (r[idx] / radius) ** (i % 4)
        else:  # Odd indices
            z_map[idx] += weight * np.sin(i * theta[idx] / 2) * (r[idx] / radius) ** (i % 4)
    
    # Normalize
    if np.max(z_map) > np.min(z_map):
        z_map = (z_map - np.min(z_map)) / (np.max(z_map) - np.min(z_map))
    
    # Save Zernike map
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}.npy")
    np.save(save_path, z_map)
    
    # Also save a visualization for quick inspection
    plt.figure(figsize=(8, 6))
    plt.imshow(z_map, cmap='viridis')
    plt.colorbar(label='Elevation')
    plt.title(f'Zernike Map: {base_name}')
    plt.tight_layout()
    
    img_save_path = os.path.join(save_dir, f"{base_name}_zernike_map.png")
    plt.savefig(img_save_path)
    plt.close()
    
    print(f"Saved Zernike map to {save_path} and visualization to {img_save_path}")
    
    return z_map