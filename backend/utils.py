import cv2, numpy as np, torch, torchvision.transforms as T
from PIL import Image

SIZE = 224
transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return tensor
