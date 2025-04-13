#generate_maps.py
import os
from zernike_utils import generate_zernike_map

input_dir = "images"
output_dir = "zernike_maps"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image file
for file in os.listdir(input_dir):
    if file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_dir, file)
        generate_zernike_map(image_path, output_dir)