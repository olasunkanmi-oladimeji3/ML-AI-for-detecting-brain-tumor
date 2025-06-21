# predict.py

import torch
from torchvision import models, transforms
from PIL import Image
import sys

# Load class names in the same order used for training
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Set up model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("brain_tumor_classifier.pt", map_location=torch.device("cpu")))
model.eval()

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image from command line
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    predicted_idx = torch.argmax(probs).item()
    print(f"🧠 Predicted: {class_names[predicted_idx]} ({probs[predicted_idx]*100:.2f}% confidence)")
