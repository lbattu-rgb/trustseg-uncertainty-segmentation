import os #Provides OS-level filesystem operations
import cv2 #OpenCV library for image processing
import numpy as np #Numerical computing library for array operations
from torch.utils.data import Dataset #PyTorch's Dataset class for creating custom datasets
import albumentations as A #Image augmentation library for data preprocessing, composable transform pipelines
from albumentations.pytorch import ToTensorV2 #Converts images to PyTorch tensors, compatible with Albumentations

class ISICDataset(Dataset):
    """Wraps the ISIC skin lesion dataset (images + segmentation masks stored as separate files on disk) 
    into the PyTorch Dataset interface, with configurable preprocessing/augmentation."""
    
    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        self.image_dir = image_dir #source directory containing input images
        self.mask_dir = mask_dir #source directory containing corresponding segmentation masks
        self.img_size = img_size #target square resoltion everything gets resized to, CNNS needs fixed-size input batches
        self.augment = augment #boolean flag indicating whether to apply data augmentation during training (True for training data, and false for validation data)

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        self.transform = self._build_transforms() #Albumentations pipeline object, built once in _build_transforms()

    def _build_transforms(self):
        """Builds the Albumentations transformation pipeline for preprocessing and augmentation."""

        if self.augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size), #force every image/mask to exactly the same size, required for batching in CNNs
                A.HorizontalFlip(p=0.5), #mirror left-right with 50% probability
                A.VerticalFlip(p=0.5), #mirror up-down with 50% probability
                A.RandomRotate90(p=0.5), #randomly rotate the image by 90 degrees (clockwise or counter-clockwise) with 50% probability
                A.ColorJitter(p=0.3), #randomly change brightness, contrast, saturation, and hue with 30% probability
                A.Normalize(mean=(0.485, 0.456, 0.406), #subtract per-channel mean and divide by per-channel standard deviation.
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2() #convert to a PyTorch tensor in CHW (Channels x Height x Width) order.
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        """PyTorch's DataLoader needs to know the total sample count to know when an epoch ends 
        and how to construct batches/shuffled indices."""
        return len(self.images)

    def __getitem__(self, idx):
        """Given an index, loads the corresponding image and segmentation mask from disk as a pair, applies preprocessing/augmentation,"""
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '_segmentation.png') #Uses ISIC dataset naming convention: if the image is named "ISIC_0000000.jpg", the corresponding mask is named "ISIC_0000000_segmentation.png"

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path) #load the raw image, OpenCV returns it in BGR order as (H, W, 3) array.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB order, since PyTorch models expect images in RGB order.

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #load mask as a single-channel grayscale image, where pixel values indicate class labels (0 for background, 255 for skin lesion).
        mask = (mask > 127).astype(np.float32) #convert mask to binary (0 or 1) float32 array, where 1 indicates lesion and 0 indicates background.

        transformed = self.transform(image=image, mask=mask) #albumentations expects a dictionary with keys 'image' and 'mask', and returns a dictionary with the same keys containing the transformed tensors.
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)

        return image, mask #(image_tensor, mask_tensor)