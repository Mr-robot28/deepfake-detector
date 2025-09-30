import timm, torch, torch.nn as nn

MODEL_NAME = 'efficientnet_b0'
DEVICE     = 'cpu'

def build_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
    in_f  = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 2)
    )
    return model.to(DEVICE).eval()
