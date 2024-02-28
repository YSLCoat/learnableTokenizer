
import torch
import torch.nn as nn
from skimage.segmentation import slic

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
        
        assert 0, image.shape

        # Apply SLIC algorithm
        segments = slic(image, n_segments=self.n_segments, compactness=self.compactness)

        return segments
