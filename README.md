# 🧠 Keratoconus Detection using Zernike Polynomials and Deep Learning

This project provides a low-cost, automated method for detecting **keratoconus** using **smartphone-based Placido ring images**. The pipeline includes Zernike polynomial decomposition, CNN-based prediction of corneal topography, and 3D visualization of the surface.

---

## 📁 Folder Structure

placido_dataset/
├── images/               # Input Placido ring eye images (JPEG/PNG)
├── zernike_maps/         # Synthetic 2D curvature maps (from Zernike moments)
├── models/
│   └── model.py          # CNN model definition
├── utils/
│   └── zernike_utils.py  # Zernike map generator
├── generate_maps.py      # Converts input images to Zernike maps
├── train.py              # Trains CNN to predict curvature maps from images
├── visualize.py          # 3D surface plot of corneal maps
└── inference.py          # Predicts map from image and evaluates keratoconus

---

## 🧪 How It Works

1. **Placido Ring Image** → Preprocessed to isolate the cornea  
2. **Zernike Moments** → Used to generate synthetic 2D curvature maps  
3. **CNN Model** → Trained to map input image → output topography  
4. **Visualization** → Plots 3D heatmap of predicted surface  
5. **Keratoconus Detection** → Uses elevation range, cone decentration, Zernike coefficients

---

## 🚀 Setup

```bash
# Create environment
conda create -n keratoconus python=3.10
conda activate keratoconus

# Install dependencies
pip install torch torchvision numpy opencv-python mahotas matplotlib pillow
```

## 🔧 Generate Zernike Maps

```bash
python generate_maps.py
```

## 🧠 Train the Model

```bash
python train.py
```

## 🔍 Predict + Visualize + Detect

```bash
python inference.py
```