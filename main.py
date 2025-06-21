# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
import torch
from PIL import Image
import io

app = FastAPI()

# CORS (for Next.js frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels (same order as training)
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("brain_tumor_classifier.pt", map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probs).item()
        confidence = float(probs[predicted_idx]) * 100

    return {
        "tumor_detected": class_names[predicted_idx] != "notumor",
        "tumor_type": class_names[predicted_idx],
        "confidence": round(confidence, 2)
    }
