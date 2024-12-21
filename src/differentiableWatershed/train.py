import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms
from torch.optim import AdamW

class ExplainedVariationLoss(nn.Module):
    def __init__(self, num_clusters, eps=1e-8):
        super(ExplainedVariationLoss, self).__init__()
        self.num_clusters = num_clusters
        self.eps = eps

    def forward(self, x, soft_assignments):
        """
        x: (B, C_in, H, W) input image
        soft_assignments: (B, C, H, W) soft cluster probabilities
        Returns:
            loss (scalar): SSR/SST
            ev_per_sample (B,): explained variation per sample
        """
        B, C_in, H, W = x.shape

        # Compute cluster means
        cluster_counts = soft_assignments.sum(dim=(2, 3))  # (B, C)
        cluster_sums = (x.unsqueeze(2) * soft_assignments.unsqueeze(1)).sum(dim=(3, 4))  # (B, C_in, C)
        cluster_means = cluster_sums / (cluster_counts.unsqueeze(1) + self.eps)  # (B, C_in, C)

        # Reconstruct each pixel from cluster means
        cluster_means_per_pixel = torch.einsum('bmk,bkhw->bmhw', cluster_means, soft_assignments)

        # Compute SST and SSR per sample
        # Keep batch dimension by summation over C,H,W only
        global_mean = x.mean(dim=(1,2,3), keepdim=True)  # (B,1,1,1)
        SST_per_sample = ((x - global_mean)**2).sum(dim=(1,2,3))  # (B,)
        SSR_per_sample = ((x - cluster_means_per_pixel)**2).sum(dim=(1,2,3))  # (B,)

        # loss = SSR/SST (averaged over batch)
        SST_total = SST_per_sample.sum() + self.eps
        SSR_total = SSR_per_sample.sum()
        loss = SSR_total / SST_total

        # Explained variation per sample: 1 - SSR/SST
        ev_per_sample = 1.0 - (SSR_per_sample / (SST_per_sample + self.eps))
        
        return loss, ev_per_sample
    
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DifferentiableVoronoiPropagation(nn.Module):
    def __init__(self, num_clusters=196, height=224, width=224, device='cpu', init_std=5.0):
        """
        A differentiable variant of Voronoi propagation.

        Changes:
        - Centroids are learnable parameters.
        - Uses soft-min approximations for cluster assignments.
        - Produces a soft assignment map (B, C, H, W) instead of a hard mask.
        - Now includes a learnable edge-detection filter (initialized as Sobel).

        Args:
            num_clusters (int): Number of clusters (centroids).
            height (int): Height of the input image.
            width (int): Width of the input image.
            device (str): 'cpu' or 'cuda'.
            init_std (float): Std for centroid initialization around a uniform grid.
        """
        super(DifferentiableVoronoiPropagation, self).__init__()

        self.C = num_clusters
        self.H = height
        self.W = width
        self.device = torch.device(device)

        # Initialize centroids as learnable parameters
        centroids = self._initialize_centroids()
        self.centroids = nn.Parameter(centroids)  # (C, 2) positions [y, x]

        self.convert_to_greyscale = torchvision.transforms.Grayscale(num_output_channels=1)

        # Hyperparameters
        self.num_iters = 20
        self.gradient_weight = 10.0
        self.color_weight = 10.0
        self.edge_exponent = 4.0
        self.alpha = 10.0  # controls softness of soft-min

        # ---------------------------
        # Learnable edge detector
        # ---------------------------
        # We want two output channels: one for grad_x, one for grad_y.
        # We'll initialize them with the Sobel filter values.
        self.edge_detector = nn.Conv2d(
            in_channels=1, 
            out_channels=2, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # Initialize weights for sobel_x and sobel_y
        # shape: (out_channels=2, in_channels=1, kernel_size=3, kernel_size=3)
        sobel_x = torch.tensor(
            [[[-1., 0., 1.],
              [-2., 0., 2.],
              [-1., 0., 1.]]]
        )  # shape (1,3,3)

        sobel_y = torch.tensor(
            [[[-1., -2., -1.],
              [ 0.,  0.,  0.],
              [ 1.,  2.,  1.]]]
        )  # shape (1,3,3)

        # Assign them to the conv layer: 
        # - first output channel is sobel_x
        # - second output channel is sobel_y
        with torch.no_grad():
            self.edge_detector.weight[0, 0].copy_(sobel_x[0])
            self.edge_detector.weight[1, 0].copy_(sobel_y[0])

    def _initialize_centroids(self):
        # Place centroids on a rough grid
        num_cols = int(math.sqrt(self.C * self.W / self.H))
        num_rows = int(math.ceil(self.C / num_cols))

        grid_spacing_y = self.H / num_rows
        grid_spacing_x = self.W / num_cols

        centroids = []
        for i in range(num_rows):
            for j in range(num_cols):
                if len(centroids) >= self.C:
                    break
                y = (i + 0.5) * grid_spacing_y
                x = (j + 0.5) * grid_spacing_x
                centroids.append([y, x])
            if len(centroids) >= self.C:
                break
        centroids = torch.tensor(centroids, device=self.device).float()  # (C, 2)
        return centroids

    def compute_gradient_map(self, x):
        """
        Applies the learnable edge detector and computes gradient magnitude.

        Args:
            x (torch.Tensor): (B,1,H,W) grayscale input

        Returns:
            grad_map (torch.Tensor): (B,1,H,W) gradient magnitude
        """
        # The edge_detector outputs 2 channels: [grad_x, grad_y].
        edges = self.edge_detector(x)  # (B,2,H,W)
        grad_x = edges[:, 0:1, :, :]
        grad_y = edges[:, 1:2, :, :]
        grad_map = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
        return grad_map

    def forward(self, x):
        """
        Forward pass:
        1. Convert to grayscale and compute gradient map.
        2. Compute initial cost map based on centroid positions.
        3. Iteratively update the cost map using a differentiable approximation.
        4. Convert final cost map to soft assignments via a softmax.

        Returns:
            soft_assignments (Tensor): (B, C, H, W) soft clustering assignments.
        """
        B, C_in, H, W = x.shape
        assert H == self.H and W == self.W, "Input size must match model initialization."

        # Convert to grayscale if needed
        if C_in == 3:
            grayscale_image = self.convert_to_greyscale(x)
        else:
            grayscale_image = x

        # 1. Compute gradient map with the learnable filter
        grad_map = self.compute_gradient_map(grayscale_image)

        # 2. Initialize distance map for each cluster
        dist_map_per_cluster = torch.full((B, self.C, H, W), float('inf'), device=self.device)
        cyx = self.centroids  # (C, 2)
        int_cy = cyx[:, 0].long().clamp(0, H-1)
        int_cx = cyx[:, 1].long().clamp(0, W-1)
        for c_idx in range(self.C):
            dist_map_per_cluster[:, c_idx, int_cy[c_idx], int_cx[c_idx]] = 0.0

        # 3. Iterative refinement using gradient + color differences
        weighted_grad_map = (grad_map ** self.edge_exponent) * self.gradient_weight  # (B,1,H,W)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for _ in range(self.num_iters):
            updates = []
            for (dy, dx) in directions:
                # Shift dist map
                shifted_dist = torch.roll(dist_map_per_cluster, shifts=(dy, dx), dims=(2, 3))

                # Color difference
                shifted_x = torch.roll(x, shifts=(dy, dx), dims=(2,3))
                color_diff = torch.abs(x - shifted_x).sum(dim=1, keepdim=True)  # (B,1,H,W)

                # Weighted cost update
                weighted_dist_update = shifted_dist + weighted_grad_map + color_diff * self.color_weight
                updates.append(weighted_dist_update)

            all_candidates = torch.cat(
                [dist_map_per_cluster.unsqueeze(0)] + [u.unsqueeze(0) for u in updates],
                dim=0
            )
            # Soft-min across candidates
            dist_map_per_cluster = -torch.logsumexp(-self.alpha * all_candidates, dim=0) / self.alpha

        # 4. Convert final cost to soft assignments
        soft_assignments = F.softmax(-dist_map_per_cluster, dim=1)  # (B, C, H, W)

        return soft_assignments


class BSDS500Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.ground_truth_dir = os.path.join(root_dir, 'ground_truth', split)
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        gt_name = os.path.join(self.ground_truth_dir, self.image_files[idx].replace('.jpg', '.mat'))
        gt_data = sio.loadmat(gt_name)
        ground_truth = gt_data['groundTruth'][0][0][0][0][1]

        segmentation = ground_truth
        
        if isinstance(segmentation, np.ndarray) and segmentation.shape == (1, 1):
            segmentation = segmentation[0, 0]
        
        segmentation = Image.fromarray(segmentation)
        segmentation = segmentation.resize((224, 224), Image.NEAREST)
        
        segmentation = np.array(segmentation, dtype=np.int64)

        segmentation = torch.tensor(segmentation, dtype=torch.long)
        
        return image
    
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

dataset_train = BSDS500Dataset(root_dir=r'D:\Data\BSDS500\data', split='train', transform=transform)

train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0)

import random
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

NUM_CLUSTERS = 196

model = DifferentiableVoronoiPropagation(NUM_CLUSTERS, device='cuda').to('cuda')

optimizer = AdamW(
        model.parameters(),
        lr=1e-3
)

loss_fn = ExplainedVariationLoss(NUM_CLUSTERS)


# Example hyperparameters
max_epochs = 10

for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    for batch_idx, source in enumerate(train_loader):
        # Move data to GPU if available
        source = source.to('cuda')

        optimizer.zero_grad()
        
        # Forward pass: model returns soft assignments (B, C, H, W)
        output = model(source)  
        
        # Compute loss and explained variation
        loss, ev_per_sample = loss_fn(source, output)
        
        #loss.requires_grad = True

        # Backward pass and parameter update
        loss.backward()
        optimizer.step()
        
        # Print explained variation per batch and per sample
        ev_batch_mean = ev_per_sample.mean().item()
        print(f"Batch {batch_idx+1}: Loss = {loss.item():.4f}, EV (batch) = {ev_batch_mean:.4f}")
        print("Per-sample EV:", ev_per_sample.tolist())
        
        