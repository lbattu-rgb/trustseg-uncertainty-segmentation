import torch #Provides the Tesnor type and core tensor operations, used for skip connections in the U-Net architecture
import torch.nn as nn #Neural network module, provides layers, loss functions, and other utilities for building neural networks

class DoubleConv(nn.Module):
    """Two convolutional layers with batch normalization and ReLU activation, followed by dropout for regularization."""
    # Inherit from nn.Module so PyTorch can register trainable layers, include
    # their parameters in model.parameters(), and automatically propagate
    # train()/eval() mode to BatchNorm and Dropout.
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super().__init__()
        self.conv = nn.Sequential( #container holding the ordered stack below
            nn.Conv2d(in_channels, out_channels, 3, padding=1), #3x3 convolution with padding=1 to preserve spatial dimensions
            nn.BatchNorm2d(out_channels), #normalizes activations across the batch dimension per channel, stabilizing and accelrating training
            nn.ReLU(inplace=True), #saves memory by performing the ReLU operation in-place, modifying the input tensor directly
            nn.Dropout2d(p=dropout_p), #randomly zeroes out entire channels with probability dropout_p during training, helping prevent overfitting (2d version))
            nn.Conv2d(out_channels, out_channels, 3, padding=1), #another 3x3 convolution, keeping the number of channels the same. Stacking two 3x3 convolutions gives receptive field of 5x5 while using fewer parameters than a single 5x5 convolution.
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Required override of placeholder function. Takes the input tensor x, passes it through stack of layers (self.conv) in __init__, and returns the result."""
        return self.conv(x)


class UNetMCDropout(nn.Module):
    """U-Net architecture with Monte Carlo Dropout for uncertainty estimation in segmentation tasks."""
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p #storing lets the model remember what dropout rate it was created with, useful for debugging

        # Encoder: the three encoder DoubleConv stages, with channel progression 3→32, 32→64, 64→128
        # More channels = more feature detectors. Instead of only looking for RGB colors, the network gradually learns hundreds of different patterns.
        self.enc1 = DoubleConv(in_channels, 32, dropout_p)
        self.enc2 = DoubleConv(32, 64, dropout_p)
        self.enc3 = DoubleConv(64, 128, dropout_p)

        #Reduces image size by half (downsampling) after each encoder stage, so that the next encoder stage can learn higher-level features at a coarser scale
        self.pool = nn.MaxPool2d(2)

        # Bottleneck: Bottom of the U-Net, where the feature maps are the smallest and the network has the most abstract representation of the input image
        # This is where the network learns global context and relationships between features
        self.bottleneck = DoubleConv(128, 256, dropout_p)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128, dropout_p)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64, dropout_p)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32, dropout_p)

        self.final = nn.Conv2d(32, out_channels, 1) #a 1×1 convolution mapping 32 channels down to out_channels (1, for binary segmentation)

    def forward(self, x): 
        """Defines the forward pass of the U-Net model. Takes an input tensor x, passes it through the encoder, bottleneck, and decoder stages, and returns the final segmentation output."""
        # Encoder
        e1 = self.enc1(x) #shape (N, 32, 256, 256)
        e2 = self.enc2(self.pool(e1)) #shape (N, 64, 128, 128), halves spacial dimensions due to pooling
        e3 = self.enc3(self.pool(e2)) #shape (N, 128, 64, 64), halves spacial dimensions again due to pooling

        # Bottleneck
        b = self.bottleneck(self.pool(e3)) #This is the point of maximum abstraction / minimum spatial resolution, the network has the most "global" view of the image here but the least precise spatial localization

        # Decoder with skip connections
        #up3 is the upsampled version of the bottleneck output, which is concatenated with the corresponding encoder output e3 to provide high-resolution features for precise localization
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1)) #unsampled output + saved encder features from the corresponding encoder stage
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1) #1×1 conv maps 256×256×32 → 256×256×1. This is the final output: one logit per pixel

    