#visualize.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(height_map, title="Corneal Topography"):
    # Ensure it's a 2D array
    height_map = np.squeeze(height_map)

    x = np.linspace(0, 1, height_map.shape[1])
    y = np.linspace(0, 1, height_map.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = height_map

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    # Make sure you have a .npy file ready at this path
    sample_map = np.load("zernike_maps/WhatsApp Image 2025-04-13 at 22.21.12.npy")
    plot_3d_surface(sample_map)