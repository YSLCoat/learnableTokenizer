import torch
import torch.nn as nn
from skimage.segmentation import slic
import numpy as np

class SLICLayer(nn.Module):
    def __init__(self, n_segments=100, compactness=10):
        super(SLICLayer, self).__init__()
        self.n_segments = n_segments
        self.compactness = compactness

    def forward(self, batch_images):
        segmented_batch = []
        for image in batch_images:
            segments = self.slic(image)
            segmented_batch.append(segments)
        return segmented_batch

    def slic(self, image):
        # Convert image from PyTorch tensor to NumPy array
        image = image.numpy()

        # Apply SLIC algorithm
        segments = slic(image, n_segments=self.n_segments, compactness=self.compactness)

        return segments

# Example usage
batch_size = 64
image_height = 224
image_width = 224
channels = 3

# Create dummy batch of images
batch_images = torch.randn(batch_size, channels, image_height, image_width)

# Initialize SLIC layer
slic_layer = SLICLayer(n_segments=100, compactness=10)

# Process batch of images using SLIC layer
segmented_batch = slic_layer(batch_images)

print(np.asarray(segmented_batch).shape)  # (224, 224)