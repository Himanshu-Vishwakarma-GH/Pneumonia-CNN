import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import gradio as gr

# Try to import pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False
    print("Warning: pytorch-grad-cam not installed. Heatmaps will not be generated.")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def get_resnet_model():
    model = models.resnet18(weights=None) # We don't need pretrained weights when loading our own
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# --- Load the Trained Model ---
# Choose which model to load ('resnet' or 'simple_cnn')
MODEL_TYPE = 'resnet'
CHECKPOINT_PATH = 'resnet_checkpoint.pth'

if MODEL_TYPE == 'resnet':
    model = get_resnet_model()
else:
    model = SimpleCNN()

model = model.to(device)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
else:
    print(f"Warning: No checkpoint found at {CHECKPOINT_PATH}. The model will use random weights until you train it.")

model.eval()

# Classes
class_names = ['NORMAL', 'PNEUMONIA']

# Transforms for inference
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_gradcam_heatmap(image_tensor, model):
    if not HAS_GRAD_CAM:
        return np.zeros((224, 224, 3))
        
    if isinstance(model, SimpleCNN):
        target_layers = [model.conv_layers[-2]]
    else:
        target_layers = [model.layer4[-1]]
        
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

def predict_xray(img):
    if img is None:
        return None, "Please upload an image"
        
    # Process image for model
    img_pil = Image.fromarray(img).convert('RGB')
    input_tensor = test_transforms(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.nn.functional.softmax(out, dim=1)[0]
        pred_class = torch.argmax(prob).item()
        confidence = prob[pred_class].item()
        
    prediction_text = f"Prediction: **{class_names[pred_class]}**\nConfidence: {confidence:.2%}"
    
    # Generate Grad-CAM visualization
    if HAS_GRAD_CAM:
        heatmap = get_gradcam_heatmap(input_tensor, model)
        float_img = np.float32(cv2.resize(img, (224, 224))) / 255
        vis = show_cam_on_image(float_img, heatmap, use_rgb=True)
    else:
        vis = cv2.resize(img, (224, 224))
        
    return vis, prediction_text

# --- Gradio UI ---
demo = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(label="Upload Chest X-Ray"),
    outputs=[
        gr.Image(label="Grad-CAM Heatmap"),
        gr.Markdown(label="Prediction Results")
    ],
    title="Pneumonia Detection with Explainable AI",
    description="Upload a chest X-ray image to detect whether it is Normal or shows signs of Pneumonia. The heatmap highlights areas the model focused on. *Note: Train the model in the notebook first to generate the checkpoint.*"
)

if __name__ == "__main__":
    demo.launch(share=False)
