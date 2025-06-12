#inference.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ResNet34Autoencoder  

def predict_corneal_topography(image_path, model_path):
    """
    Predict corneal topography from an image using the trained model
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model weights
    
    Returns:
        Predicted topography map as numpy array
    """
    # Load model
    model = ResNet34Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
    model.eval()
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Convert to numpy for visualization
    prediction_map = prediction.squeeze().cpu().numpy()
    
    return prediction_map

def visualize_prediction(prediction_map, save_path=None):
    """
    Visualize the predicted topography map
    
    Args:
        prediction_map: The predicted map as numpy array
        save_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # 2D visualization
    plt.subplot(1, 2, 1)
    plt.imshow(prediction_map, cmap='viridis')
    plt.colorbar(label='Elevation')
    plt.title('Predicted Corneal Topography (2D)')
    
    # 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(1, 2, 2, projection='3d')
    x = np.linspace(0, 1, prediction_map.shape[1])
    y = np.linspace(0, 1, prediction_map.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, prediction_map, cmap='viridis', edgecolor='none', alpha=0.8)
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation')
    ax.set_title('Predicted Corneal Topography (3D)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "images/test/WhatsApp Image 2025-04-13 at 22.21.27 (1).jpeg"  # Replace with your test image path
    model_path = "models/corneal_model.pth"  # Path to your trained model
    
    # Make prediction
    prediction = predict_corneal_topography(image_path, model_path)
    
    # Visualize results
    visualize_prediction(prediction, save_path="predictions/prediction_result.png")
    
    # Check for keratoconus indicators
    # This is a simplified approach - in real applications you'd want a more sophisticated analysis
    max_elevation = np.max(prediction)
    min_elevation = np.min(prediction)
    elevation_range = max_elevation - min_elevation
    
    print(f"Max elevation: {max_elevation:.4f}")
    print(f"Min elevation: {min_elevation:.4f}")
    print(f"Elevation range: {elevation_range:.4f}")
    
    # Simple threshold-based detection (this is just an example)
    # You should develop more sophisticated criteria based on clinical knowledge
    if elevation_range > 0.4:
        print("Likely keratoconus")
    elif elevation_range > 0.3:
        print("Possibly early keratoconus - recommend further examination.")
    else:
        print("No obvious keratoconus indicators detected.")