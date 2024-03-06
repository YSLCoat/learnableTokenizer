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
    
    
class AttentionSpatialTransformer(nn.Module):
    def __init__(self, n_channels=3) -> None:
        super(AttentionSpatialTransformer, self).__init__()
        self.n_channels = n_channels
        self.localization = nn.Sequential(nn.Conv2d(self.n_channels, 16, kernel_size=7),
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
                                          nn.ReLU(True))

        self.fc_loc = nn.Sequential(nn.Linear(12800, 32),
                                    nn.ReLU(True),
                                    nn.Linear(32, 3))

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([0.2, 0, 0], dtype=torch.float)) #set scaling to start at 0.2 (~28/128)
 
    def forward(self: object, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        xs = xs.view(-1, 12800)
        theta = self.fc_loc(xs)

        translation = theta[:, 1:].unsqueeze(2)
        scale = theta[:, 0].unsqueeze(1)
        scale_mat = torch.cat((scale, scale), 1)
        theta = torch.cat((torch.diag_embed(scale_mat), translation), 2)

        grid = F.affine_grid(theta, torch.Size([x.size()[0], x.size()[1], 28, 28])) # downsampling from 128x128 to 28x28
        x = F.grid_sample(x, grid)
        
        return x#, theta