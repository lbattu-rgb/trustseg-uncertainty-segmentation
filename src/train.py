import torch #Training loop touches a torch.* call — device selection, gradient control, optimizer construction, checkpoint saving
from torch.utils.data import DataLoader, random_split #handles batching, shuffling, and iteration
from src.dataset import ISICDataset #data source
from src.model import UNetMCDropout #UNet model with Monte Carlo Dropout for uncertainty estimation
import os

def dice_loss(pred, target, smooth=1):
    """Standard Dice loss function for binary segmentation tasks. Takes predicted logits and ground truth masks, applies sigmoid to predictions, and computes the Dice coefficient."""
    pred = torch.sigmoid(pred) #converts raw digits into probabilities between 0 and 1, which is necessary for the Dice loss calculation
    intersection = (pred * target).sum(dim=(2, 3)) #"masks out" prediction values at background pixels and keeps only the predicted probability values at pixels that are actually lesion
    dice = (2 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth) #Dice coefficient formula: 2 * |A ∩ B| / (|A| + |B|), with smoothing to avoid division by zero
    return 1 - dice.mean() #Optimizers minimize loss, so return 1 - Dice coefficient to turn it into a loss function (lower is better).

def train():
    """Main training loop for the U-Net model with Monte Carlo Dropout. Loads the dataset, splits into training and validation sets, defines the model, optimizer, and learning rate scheduler, and iteratively trains the model while tracking validation loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ISICDataset( #builds the full dataset with augmentation enabled, inherits augument = True from the ISICDataset class
        image_dir="data/images",
        mask_dir="data/masks",
        img_size=256,
        augment=True
    )

    train_size = int(0.8 * len(dataset)) #computes an 80/20 split by count by convention
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size]) #randomly partitions indices into the two subsets

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True) #Takes the training dataset, randomly shuffles it at the start of each epoch, and feeds it to the model 8 images at a time
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False) #same batch size, but no shuffling needed, since validation doesn't update weights and processing order therefore doesn't affect anything

    model = UNetMCDropout(dropout_p=0.3).to(device) #instantiate the model and move all its parameters to the selected device
    
    #Optimizer updates the model's weights based on the gradients computed during backpropagation. 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #Adam automatically figures out how much each weight should change
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=5, #number of epochs with no improvement after which learning rate will be reduced
        factor=0.5 #reduces the learning rate by this factor when triggered
        )   

    best_val_loss = float('inf') #initialize a running "best score so far" tracker to positive infinity, so always start with improvement on the first epoch
    epochs = 50 #total number of passes over the training set

    for epoch in range(epochs):
        model.train() #switch into training mode, enabling dropout and batch normalization updates
        train_loss = 0 #sum accumulator for this epoch's training loss
        for images, masks in train_loader: # Loop through each training batch
            images, masks = images.to(device), masks.to(device) # Move data to the CPU or GPU
            optimizer.zero_grad() # Clear old gradients from the previous batch
            outputs = model(images) # Run the images through the model to make predictions
            loss = dice_loss(outputs, masks) # Compare predictions to the true masks
            loss.backward() # Calculate how each weight contributed to the loss
            optimizer.step() # Update the model's weights using the optimizer
            train_loss += loss.item() # Add this batch's loss to the total training loss

        model.eval() # Put the model into evaluation mode
        val_loss = 0 # Keep track of the total validation loss
        with torch.no_grad(): # Turn off gradient calculations to save memory and speed up validation
            for images, masks in val_loader: 
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, masks)
                val_loss += loss.item()

        train_loss /= len(train_loader) # Calculate the average training loss
        val_loss /= len(val_loader) # Calculate the average validation loss
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}") # Display training progress
        scheduler.step(val_loss) # Reduce the learning rate if the validation loss has stopped improving

        if val_loss < best_val_loss: # Check if this is the best validation loss so far
            best_val_loss = val_loss # Save the new best validation loss
            os.makedirs("model", exist_ok=True) # Create the model folder if it doesn't already exist
            torch.save(model.state_dict(), "model/best_model.pth") # Save the model's learned weights
            print("  Saved best model!")

if __name__ == "__main__":
    train() # Start training the model