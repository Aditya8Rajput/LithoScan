import os
import cv2
import base64
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI(title="LithoScan Diagnostic Command Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LithoScan Engine Active on: {device}")

# Load Trained ResNet50 Model
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('lithoscan_resnet50_best.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Ensure gradients are enabled on layer4 for Grad-CAM
for param in model.layer4.parameters():
    param.requires_grad = True

classes = ['Normal', 'Stone']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def encode_rgb_to_base64(rgb_img):
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', bgr_img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/confusion-matrix")
async def get_confusion_matrix():
    if os.path.exists("confusion_matrix.png"):
        return FileResponse("confusion_matrix.png")
    return {"error": "confusion_matrix.png not found"}

@app.post("/predict")
async def predict_scan(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # 1. Raw & CLAHE Processing
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    raw_rgb = cv2.cvtColor(cv2.resize(img_gray, (224, 224)), cv2.COLOR_GRAY2RGB)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    clahe_resized = cv2.resize(clahe_rgb, (224, 224))
    
    input_tensor = transform(Image.fromarray(clahe_rgb)).unsqueeze(0).to(device)
    
    # 2. Model Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probs, 0)
        
    pred_class = predicted_idx.item()
    normal_prob = round(probs[0].item() * 100, 2)
    stone_prob = round(probs[1].item() * 100, 2)

    # 3. Generate Grad-CAM Heatmap
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    rgb_float = clahe_resized / 255.0
    gradcam_rgb = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return {
        "filename": file.filename,
        "prediction": classes[pred_class],
        "confidence": round(confidence.item() * 100, 2),
        "normal_prob": normal_prob,
        "stone_prob": stone_prob,
        "raw_image": encode_rgb_to_base64(raw_rgb),
        "clahe_image": encode_rgb_to_base64(clahe_resized),
        "gradcam_image": encode_rgb_to_base64(gradcam_rgb)
    }