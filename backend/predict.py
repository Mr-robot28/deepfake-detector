import torch, pathlib
from backend.model import build_model
from backend.utils import preprocess

model = build_model()

def predict_image(path):
    with torch.no_grad():
        tensor = preprocess(path)
        logits = model(tensor)
        prob   = torch.softmax(logits, dim=1)[0].tolist()
    return {"real": prob[0], "fake": prob[1]}
