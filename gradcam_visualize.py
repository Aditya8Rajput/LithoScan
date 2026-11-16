import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 1. Setup Device & Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Grad-CAM on: {device}")

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('lithoscan_resnet50_best.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# 2. Grab a Sample "Stone" Image
stone_dir = r"D:\New Projects\LithoScan\archive (4)\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\Stone"
sample_img_name = os.listdir(stone_dir)[0]  # Grabs the very first stone scan
img_path = os.path.join(stone_dir, sample_img_name)

# 3. Apply the exact same CLAHE preprocessing
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_gray)
img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

# 4. Transform for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

# 5. Grad-CAM Setup targeting the final layer of ResNet50
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Target the "Stone" class (which is index 1 based on our earlier sorting)
targets = [ClassifierOutputTarget(1)]

# Generate the heatmap
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

# 6. Overlay Heatmap on the Original Image
rgb_img_float = cv2.resize(img_rgb, (224, 224)) / 255.0
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

# 7. Save the Output
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original CLAHE CT Scan")
plt.imshow(rgb_img_float)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("AI Focus Heatmap (Grad-CAM)")
plt.imshow(visualization)
plt.axis('off')

plt.savefig('gradcam_result.png', dpi=300, bbox_inches='tight')
print("Success! Check your folder for 'gradcam_result.png'")