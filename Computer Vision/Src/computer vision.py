#================= Importance ===================
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from timm import create_model
from sklearn.model_selection import train_test_split
import timm

#============= Configeration =======================

Xception_path ="/kaggle/working/best_xception.pth"
YOLO_path ="/kaggle/input/yolo_hand/pytorch/default/1/yolov8_trained.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=========== Modeling ===============================
class BoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=True, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x).squeeze()

model = BoneAgeModel()
state_dict = torch.load(Xception_path, map_location=device)
model.load_state_dict(state_dict)  
model = model.to(device)

yolo_model = YOLO(YOLO_path)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def how_old_are_you(img):
    
    # img = Image.open(image_path).convert("RGB")
    

    
    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print("❌ لم يتم اكتشاف يد في الصورة")
        return None

    
    x1, y1, x2, y2 = boxes[0].astype(int)
    cropped = img.crop((x1, y1, x2, y2))

    # تطبيق الـ transforms
    input_tensor = transform(cropped).unsqueeze(0).to(device)

    # توقع العمر
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).item()

    return pred , cropped