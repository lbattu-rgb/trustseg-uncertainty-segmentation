import torch
import numpy as np

def enable_dropout(model):
    """Enables dropout layers during inference for Monte Carlo Dropout. This allows the model to produce different outputs for the same input, which can be used to estimate uncertainty."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train() #flips the dropout layers into training mode, so they will randomly zero out channels during inference, allowing for stochastic forward passes

def mc_predict(model, image_tensor, n_passes=20, device='cpu'):
    """Performs Monte Carlo Dropout predictions on a single image tensor. Runs the model multiple times with dropout enabled, collects the predictions, and computes the mean and variance to estimate uncertainty."""
    model.eval()
    enable_dropout(model) #re-enable just the dropout layers while keeping batch norm in its stable eval-mode behavior
    
    image_tensor = image_tensor.unsqueeze(0).to(device) #add a batch dimension (N=1) and move the tensor to the specified device (CPU or GPU)
    predictions = [] #collect each pass's output

    with torch.no_grad(): #disable gradient tracking for the entire loop, since this is pure inference, not training
        for _ in range(n_passes): #loop 20 times to get 20 different predictions for the same input image
            output = model(image_tensor) #one full forward pass through the network, the exact same input tensor produces a slightly different output each time
            prob = torch.sigmoid(output) #convert the raw logit output into per-pixel probabilities in [0, 1]
            predictions.append(prob.cpu().numpy()) #move the tensor back to CPU and convert to a NumPy array for easier manipulation and storage

    predictions = np.array(predictions) #stack all 20 indvidual predictions into a single NumPy array of shape (n_passes, 1, H, W)

    mean_pred = predictions.mean(axis=0).squeeze() #average across the "pass" axis (axis 0), collapsing 20 different predictions into one averaged probability map
    uncertainty = predictions.var(axis=0).squeeze() #compute the variance across the "pass" axis (axis 0), giving a per-pixel measure of how much the predictions varied across the 20 passes. High variance indicates high uncertainty in the model's prediction for that pixel.

    return mean_pred, uncertainty 


def dice_score(pred, mask, threshold=0.5):
    """Accuracy metric for reporting how good a prediction is against ground truth """
    pred_binary = (pred > threshold).astype(np.float32) #threshold the continuous probability map into a hard binary mask (0 or 1)
    intersection = (pred_binary * mask).sum() #count of pixels where both prediction and ground truth agree the pixel is lesion
    return (2 * intersection + 1) / (pred_binary.sum() + mask.sum() + 1) #Dice coefficient formula with smoothing to avoid division by zero. Returns a value in [0, 1], where 1 is perfect overlap and 0 is no overlap.