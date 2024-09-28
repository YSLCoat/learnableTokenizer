import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class spatialTransformer(nn.Module):
    def __init__(self, n_channels=1):
        super(spatialTransformer, self,).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(self.n_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.n_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        return x
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSpatialTransformer(nn.Module):
    def __init__(self, output_dim, n_channels=3) -> None:
        super(AttentionSpatialTransformer, self).__init__()
        self.n_channels = n_channels
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Initialize fc_loc in forward based on input size
        self.fc_loc = None

        # Final linear layer
        self.fc_final = nn.Linear(3 * 28 * 28, output_dim)

    def initialize_fc_loc(self, xs: torch.Tensor):
        """
        Initialize the fully connected layer dynamically based on the flattened input size.
        """
        flattened_size = xs.size(1) * xs.size(2) * xs.size(3)
        self.fc_loc = nn.Sequential(
            nn.Linear(flattened_size, 32),  # Dynamically adjust the input size
            nn.ReLU(True),
            nn.Linear(32, 3)  # 3 parameters: scaling and translation
        ).to(xs.device)
        # Initialize weights for the affine transformation
        self.fc_loc[2].weight.data.zero_().to(xs.device)
        self.fc_loc[2].bias.data.zero_().to(xs.device)
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.2, 0, 0], dtype=torch.float)).to(xs.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        height, width = x.size(2), x.size(3)
        
        # Apply the localization network
        xs = self.localization(x)
        
        # Initialize fc_loc based on the size of xs, if not initialized yet
        if self.fc_loc is None:
            self.initialize_fc_loc(xs)
        
        # Flatten the tensor for the fully connected layer
        xs_flattened_size = xs.size(1) * xs.size(2) * xs.size(3)
        xs = xs.view(-1, xs_flattened_size)  # Flatten for fully connected layers
        
        # Move xs to the same device as input x
        xs = xs.to(x.device)

        # Pass through the fully connected localization layers
        theta = self.fc_loc(xs)

        # Extract scaling and translation parameters
        translation = theta[:, 1:].unsqueeze(2).to(x.device)  # Ensure translation is on the same device
        scale = theta[:, 0].unsqueeze(1).to(x.device)  # Ensure scale is on the same device
        scale_mat = torch.cat((scale, scale), 1)  # Create a 2D scale matrix
        
        # Create the affine transformation matrix
        theta = torch.cat((torch.diag_embed(scale_mat), translation), 2)

        # Adjust the target size of the grid based on input dimensions
        target_height, target_width = 28, 28  # Target downsampling size (this can be changed)
        grid = F.affine_grid(theta, torch.Size([batch_size, self.n_channels, target_height, target_width]), align_corners=False)
        
        # Apply the affine transformation
        x = F.grid_sample(x, grid, align_corners=False)
        
        # Flatten the output and pass through the final linear layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_final(x)  # Linear layer to get the desired output dimension
        
        return x
