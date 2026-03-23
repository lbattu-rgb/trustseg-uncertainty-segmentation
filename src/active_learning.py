import torch
import numpy as np
from src.uncertainty import mc_predict
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess(image):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    image_np = np.array(image.convert("RGB"))
    return transform(image=image_np)['image']

def rank_by_uncertainty(model, images, device, n_passes=10):
    results = []

    for name, image in images:
        tensor = preprocess(image)
        mean_pred, uncertainty = mc_predict(
            model, tensor, n_passes=n_passes, device=device
        )
        avg_uncertainty = float(uncertainty.mean())
        results.append({
            "name": name,
            "image": image,
            "uncertainty": avg_uncertainty,
            "mean_pred": mean_pred,
            "uncertainty_map": uncertainty
        })

    results.sort(key=lambda x: x["uncertainty"], reverse=True)
    return results