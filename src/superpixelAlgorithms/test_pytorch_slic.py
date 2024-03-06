import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import litdata.litdata as litdata

postprocess = (
    torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    nn.Identity(),
    )


train_dataset = litdata.LITDataset(
    r"IN1k",
    r"F:\data",
    override_extensions=[
        'jpg',
        'cls'
    ],
    train = True,
).map_tuple(*postprocess)

val_dataset = litdata.LITDataset(
    r"IN1k",
    r"F:\data",
    override_extensions=[
        'jpg',
        'cls'
    ],
    train = False,
).map_tuple(*postprocess)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
# Assuming you have your dataset and dataloader set up
image, _ = train_dataset[3]

class SLICLayer(nn.Module):
    def __init__(self, image_size, num_segments, compactness):
        super(SLICLayer, self).__init__()
        self.image_size = image_size
        self.num_segments = num_segments
        self.compactness = compactness

        # Initialize cluster centers
        self.cluster_centers = nn.Parameter(torch.rand(num_segments, 2) * image_size, requires_grad=True)

    def forward(self, image):
        batch_size, num_channels, height, width = image.size()

        # Flatten image and compute feature vectors
        image_flat = image.view(batch_size, num_channels, -1)  # Shape: (batch_size, num_channels, height * width)
        features = image_flat.permute(0, 2, 1)  # Shape: (batch_size, height * width, num_channels)

        # Initialize segmentation map
        segmentation_map = torch.zeros(batch_size, height * width, dtype=torch.long, device=image.device)

        # Iterate until convergence (or for a fixed number of iterations)
        for _ in range(num_iterations):
            # Assign pixels to nearest cluster centers
            self.assign_pixels(features, segmentation_map)

            # Update cluster centers
            self.update_centers(image, features, segmentation_map)

        # Reshape segmentation map to match image dimensions
        segmentation_map = segmentation_map.view(batch_size, height, width)

        return segmentation_map

    def assign_pixels(self, features, segmentation_map):
        batch_size, num_pixels, num_channels = features.size()
        _, num_segments, _ = self.cluster_centers.size()

        for i in range(batch_size):
            for j in range(num_pixels):
                pixel = features[i, j]  # Current pixel feature vector
                distances = torch.norm(pixel.unsqueeze(0) - self.cluster_centers, dim=2)  # Compute distances to cluster centers
                nearest_cluster = torch.argmin(distances)  # Find index of nearest cluster center
                segmentation_map[i, j] = nearest_cluster  # Assign pixel to nearest cluster

    def update_centers(self, image, features, segmentation_map):
        batch_size, num_pixels, num_channels = features.size()
        _, num_segments, _ = self.cluster_centers.size()

        # Compute mean feature vector and position for each cluster
        for i in range(batch_size):
            for k in range(num_segments):
                # Find pixels assigned to current cluster
                cluster_pixels = features[i, segmentation_map[i] == k]

                # Compute mean feature vector
                if len(cluster_pixels) > 0:
                    mean_feature = torch.mean(cluster_pixels, dim=0)
                else:
                    mean_feature = torch.zeros(num_channels, device=image.device)

                # Compute mean position
                positions = torch.nonzero(segmentation_map[i] == k)
                mean_position = torch.mean(positions.float(), dim=0)

                # Update cluster center
                self.cluster_centers[:, k] = mean_position

# Example usage
image_size = 224*224#(224, 224)
num_segments = 100
compactness = 10
num_iterations = 10

# Create SLIC layer
slic_layer = SLICLayer(image_size, num_segments, compactness)

# Generate random input image (replace this with actual image data)
#image = torch.rand(1, 3, *image_size)

# Apply SLIC segmentation
segmentation_map = slic_layer(image.unsqueeze(0))

# Print segmentation map shape
print("Segmentation map shape:", segmentation_map.shape)