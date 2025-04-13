#train.py
import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from model import CornealTopographyCNN  # Fixed import path
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for Placido images and corresponding Zernike maps
class PlacidoDataset(Dataset):
    def __init__(self, image_dir, map_dir):
        self.image_dir = image_dir
        self.map_dir = map_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Handle different possible file extensions
        base_name = os.path.splitext(img_name)[0]
        map_path = os.path.join(self.map_dir, f"{base_name}.npy")
        
        image = Image.open(img_path)
        image = self.transform(image)
        
        try:
            z_map = torch.tensor(np.load(map_path), dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]
        except FileNotFoundError:
            print(f"Warning: Zernike map not found for {img_name} at {map_path}")
            # Create a dummy map or raise exception based on your preference
            z_map = torch.zeros((1, 256, 256), dtype=torch.float32)
            
        return image, z_map

# Training function
def train():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    dataset = PlacidoDataset("images", "zernike_maps")
    
    if len(dataset) == 0:
        print("Error: No images found in dataset directory. Please add images to placido_dataset/images.")
        return
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = CornealTopographyCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, maps) in enumerate(dataloader):
            images, maps = images.to(device), maps.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, maps)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    # Save model weights
    torch.save(model.state_dict(), "models/corneal_model.pth")
    print("✅ Model saved to placido_dataset/models/corneal_model.pth")

if __name__ == "__main__":
    train()