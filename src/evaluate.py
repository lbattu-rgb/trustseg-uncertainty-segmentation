import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import ISICDataset
from src.model import UNetMCDropout
from src.uncertainty import mc_predict, dice_score
#mc_predict calls model.eval(), encapsulated in that function

def evaluate():
    """The single entry point tying together the dataset, the trained model, and the uncertainty/accuracy measurement tools, run across the entire dataset to produce aggregate statistics and a visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ISICDataset( #build the dataset with augmentation off, since evaluation needs deterministic, representative preprocessing to produce meaningful, comparable Dice scores
        image_dir="data/images",
        mask_dir="data/masks",
        img_size=256,
        augment=False
    )

    model = UNetMCDropout(dropout_p=0.3).to(device) #reconstruct an empty model with the same architecture used during training
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device)) #load the trained weights from disk into the freshly constructed mode

    dice_scores = []
    uncertainties = []

    for i in range(len(dataset)): #loop through every image in dataset
        image, mask = dataset[i] #get the preprocessed image tensor and ground-truth mask tensor for this index
        mask_np = mask.squeeze().numpy()

        mean_pred, uncertainty = mc_predict(model, image, n_passes=20, device=device) #run the full 20-pass Monte Carlo Dropout inference for this single image

        dice = dice_score(mean_pred, mask_np) #compute how accurate this prediction is against ground truth
        avg_uncertainty = uncertainty.mean()#collapse the per-pixel uncertainty map into one scalar summary number for this image

        dice_scores.append(dice)
        uncertainties.append(avg_uncertainty)

        if i % 20 == 0: #print progress line every 20 images for user 
            print(f"[{i}/{len(dataset)}] Dice: {dice:.4f} | Uncertainty: {avg_uncertainty:.6f}")

#Convert into NumPy arrays for easier plotting and statistics
    dice_scores = np.array(dice_scores)
    uncertainties = np.array(uncertainties)

    plt.figure(figsize=(8, 5))
    plt.scatter(uncertainties, dice_scores, alpha=0.6, color='steelblue')
    plt.xlabel("Average Uncertainty (Variance)")
    plt.ylabel("Dice Score")
    plt.title("Uncertainty vs Segmentation Performance")
    plt.savefig("uncertainty_vs_dice.png", dpi=150) #save figure to disk with at a specfic resolution (dots-per-inch)
    plt.close()

    print(f"\nMean Dice: {dice_scores.mean():.4f}")
    print(f"Mean Uncertainty: {uncertainties.mean():.6f}")
    print("Saved plot to uncertainty_vs_dice.png")

if __name__ == "__main__":
    evaluate()