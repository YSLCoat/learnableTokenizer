import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplainedVariationLoss(nn.Module):
    def __init__(self, num_clusters, eps=1e-8):
        super(ExplainedVariationLoss, self).__init__()
        self.num_clusters = num_clusters
        self.eps = eps

    def forward(self, x, soft_assignments):
        """
        x: (B, C_in, H, W) input image
        soft_assignments: (B, C, H, W) soft cluster probabilities
        """
        B, C_in, H, W = x.shape

        # Compute cluster counts and cluster means
        cluster_counts = soft_assignments.sum(dim=(2, 3))  # (B, C)
        cluster_sums = (x.unsqueeze(2) * soft_assignments.unsqueeze(1)).sum(dim=(3, 4))  # (B, C_in, C)
        cluster_means = cluster_sums / (cluster_counts.unsqueeze(1) + self.eps)  # (B, C_in, C)

        # Reconstruct the pixel values using cluster means weighted by soft assignments
        cluster_means_per_pixel = torch.einsum('bci,bchw->bchw', cluster_means, soft_assignments)

        # Compute explained variation loss
        global_mean = x.mean()
        SST = ((x - global_mean) ** 2).sum()
        SSR = ((x - cluster_means_per_pixel) ** 2).sum()

        loss = SSR / (SST + self.eps)
        return loss