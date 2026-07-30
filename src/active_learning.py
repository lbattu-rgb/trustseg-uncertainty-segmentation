import numpy as np
from src.uncertainty import mc_predict
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess(image):
    """Preprocesses a PIL image for model input. Resizes, normalizes, and converts to a PyTorch tensor."""
    transform = A.Compose([
        A.Resize(256, 256), # Resize the image to the size expected by the model
        A.Normalize(mean=(0.485, 0.456, 0.406), # Normalize pixel values using ImageNet statistics
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2() # Convert the image into a PyTorch tensor
    ])
    image_np = np.array(image.convert("RGB")) # Convert the PIL image to a NumPy RGB image
    return transform(image=image_np)['image'] # Apply preprocessing and return the processed image

def rank_by_uncertainty(model, images, device, n_passes=10): 
    """Ranks a list of images by the model's uncertainty in its predictions. Returns a sorted list of dictionaries containing image names, mean predictions, and uncertainty scores."""
    results = [] # Store the prediction results for each image

    for name, image in images: 
        tensor = preprocess(image) # Preprocess the image for the model
        mean_pred, uncertainty = mc_predict( # Predict the segmentation and uncertainty
            model, tensor, n_passes=n_passes, device=device
        )
        avg_uncertainty = float(uncertainty.mean()) # Compute one uncertainty score for the whole image
        results.append({ # Save all results for this image
            "name": name,
            "image": image,
            "uncertainty": avg_uncertainty,
            "mean_pred": mean_pred,
            "uncertainty_map": uncertainty
        })

    results.sort(key=lambda x: x["uncertainty"], reverse=True) # Sort images from highest uncertainty to lowest
    return results 