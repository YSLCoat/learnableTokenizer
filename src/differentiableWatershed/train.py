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
    
    
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class HardUpdateMask(Function):
    """
    The HardUpdateMask uses a straight‐through estimator to backpropagate through a hard comparison. 
    In the forward pass, it performs (new_dist < old_dist).float()—a non‐differentiable step function—to produce a 0/1 mask. 
    However, in the backward pass it “pretends” this mask is a continuous gating factor, passing the gradient to new_dist wherever the mask is 1,
    and to old_dist wherever the mask is 0. This is not the true mathematical derivative of a step function, 
    but it preserves a gradient path through discrete assignments, allowing upstream parameters (like a learnable edge filter) to be updated.
    """
    
    
    @staticmethod
    def forward(ctx, new_dist, old_dist):
        """
        Args:
          new_dist:  (B,H,W) proposed new distance
          old_dist:  (B,H,W) existing distance
        Returns:
          update_mask: (B,H,W) boolean (float 0 or 1) telling where new_dist < old_dist
        """
        update_mask = (new_dist < old_dist).float()  # Hard comparison
        ctx.save_for_backward(update_mask)
        return update_mask

    @staticmethod
    def backward(ctx, grad_output):
        """
        We must return gradients w.r.t. new_dist and old_dist, the same shape as the inputs in forward.
        
        grad_output has shape (B,H,W), which is the gradient wrt update_mask from downstream.
        """
        (update_mask,) = ctx.saved_tensors  # shape (B,H,W)

        # Straight-through: pass the gradient to new_dist where update_mask=1,
        # and pass the gradient to old_dist where update_mask=0.
        grad_new_dist = grad_output * update_mask
        grad_old_dist = grad_output * (1 - update_mask)

        return grad_new_dist, grad_old_dist
    
    
class HardFinalMask(Function):
    """
    Straight-through for the *final* mask assignment.
    Forward: returns updated_mask (B,H,W) with integer cluster IDs.
    Backward: routes grad to the new_dist where the update happened.
    """

    @staticmethod
    def forward(ctx, new_dist, old_dist, shifted_mask, old_mask):
        """
        new_dist:    (B,H,W) proposed new distance
        old_dist:    (B,H,W) existing distance
        shifted_mask:(B,H,W) the cluster IDs from the neighbor
        old_mask:    (B,H,W) the existing cluster IDs

        Returns:
          final_mask: (B,H,W) integer cluster IDs after update
        """
        update_mask = (new_dist < old_dist)  # boolean
        final_mask = torch.where(update_mask, shifted_mask, old_mask)
        # store update_mask for backward
        ctx.save_for_backward(update_mask.float())
        return final_mask  # shape (B,H,W), integer

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: gradient wrt final_mask, shape (B,H,W).
        We return 4 grads for new_dist, old_dist, shifted_mask, old_mask, 
        matching the forward signature.
        """
        (update_mask,) = ctx.saved_tensors  # shape (B,H,W) float

        # We don't have a direct gradient route for the integer mask w.r.t. cluster IDs,
        # but we do have a route to new_dist or old_dist. 
        # Let's do a simple STE: pass grad to new_dist if update_mask=1, else old_dist.
        grad_new_dist = grad_output * update_mask
        grad_old_dist = grad_output * (1. - update_mask)

        # No gradient flows to shifted_mask or old_mask (they're integer cluster IDs).
        grad_shifted_mask = None
        grad_old_mask = None

        return grad_new_dist, grad_old_dist, grad_shifted_mask, grad_old_mask
    
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VoronoiPropagation(nn.Module):
    def __init__(self, num_clusters=196, height=224, width=224, device='cpu'):
        super(VoronoiPropagation, self).__init__()
        
        self.C = num_clusters
        self.H = height
        self.W = width
        self.device = torch.device(device)
        
        #    We interpret channel 0 as Gx, channel 1 as Gy (loosely).
        self.edge_filter = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        # Initialize approximate Sobel
        sobel_weights = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]],   # Gx
             [[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]],  # Gy
        ).float().unsqueeze(1)  # shape (2,1,3,3)
        with torch.no_grad():
            self.edge_filter.weight.copy_(sobel_weights)

        self.convert_to_greyscale = torchvision.transforms.Grayscale(num_output_channels=1)

    def compute_gradient_map(self, x):
        """
        Now uses our trainable convolution to produce Gx, Gy,
        then returns gradient magnitude = sqrt(Gx^2 + Gy^2).
        """
        # x is (B,1,H,W) if grayscale
        edge_out = self.edge_filter(x)  # shape (B,2,H,W)
        gx = edge_out[:, 0:1, :, :]
        gy = edge_out[:, 1:2, :, :]
        grad_map = torch.sqrt(gx.pow(2) + gy.pow(2) + 1.0e-8)  # small eps for stability
        return grad_map

    def place_centroids_on_grid(self, batch_size):
        num_cols = int(math.sqrt(self.C * self.W / self.H))
        num_rows = int(math.ceil(self.C / num_cols))

        grid_spacing_y = self.H / num_rows
        grid_spacing_x = self.W / num_cols

        centroids = []
        for i in range(num_rows):
            for j in range(num_cols):
                if len(centroids) >= self.C:
                    break
                y = int((i + 0.5) * grid_spacing_y)
                x = int((j + 0.5) * grid_spacing_x)
                centroids.append([y, x])
            if len(centroids) >= self.C:
                break

        centroids = torch.tensor(centroids, device=self.device).float()
        return centroids.unsqueeze(0).repeat(batch_size, 1, 1)

    def find_nearest_minima(self, centroids, grad_map, neighborhood_size=10):
        """
        Still does a discrete search for local minima—this breaks gradient by default,
        but we either accept no gradient here or do a bigger STE. 
        For brevity, let's leave it as-is. 
        """
        updated_centroids = []
        B, _, _ = centroids.shape
        
        for batch_idx in range(B):
            updated_centroids_batch = []
            occupied_positions = set()
            for centroid in centroids[batch_idx]:
                y, x = centroid
                y_min = max(0, int(y) - neighborhood_size)
                y_max = min(self.H, int(y) + neighborhood_size)
                x_min = max(0, int(x) - neighborhood_size)
                x_max = min(self.W, int(x) + neighborhood_size)
                
                neighborhood = grad_map[batch_idx, 0, y_min:y_max, x_min:x_max]
                min_val = torch.min(neighborhood)
                min_coords = torch.nonzero(neighborhood == min_val, as_tuple=False)
                
                found = False
                for coord in min_coords:
                    new_y = y_min + coord[0].item()
                    new_x = x_min + coord[1].item()
                    position = (new_y, new_x)
                    if position not in occupied_positions:
                        occupied_positions.add(position)
                        updated_centroids_batch.append([new_y, new_x])
                        found = True
                        break
                if not found:
                    updated_centroids_batch.append([y.item(), x.item()])
            
            updated_centroids.append(torch.tensor(updated_centroids_batch, device=self.device))
        
        return torch.stack(updated_centroids, dim=0)

    def distance_weighted_propagation(
        self, centroids, grad_map, color_map,
        num_iters=50, gradient_weight=10.0,
        color_weight=10.0, edge_exponent=4.0
    ):
        """
        Hard Voronoi-like propagation, but we wrap the 'update_mask = weighted_dist < dist_map'
        step in our HardUpdateMask STE so that some gradient can still flow backward.
        """
        B, _, H, W = grad_map.shape
        mask = torch.full((B, H, W), fill_value=-1, device=grad_map.device)
        dist_map = torch.full((B, H, W), fill_value=float('inf'), device=grad_map.device)
        
        # Initialize distance=0 at each centroid
        for batch_idx in range(B):
            for idx, (cy, cx) in enumerate(centroids[batch_idx]):
                mask[batch_idx, int(cy), int(cx)] = idx
                dist_map[batch_idx, int(cy), int(cx)] = 0
        
        # 4-connected neighbors (dy, dx)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Weighted gradient map
        weighted_grad_map = (grad_map ** edge_exponent) * gradient_weight  # shape (B,1,H,W)

        for _ in range(num_iters):
            for dy, dx in directions:
                shifted_dist = torch.roll(dist_map, shifts=(dy, dx), dims=(1, 2))
                shifted_mask = torch.roll(mask, shifts=(dy, dx), dims=(1, 2))
                
                # color distance between neighbors
                color_diff = torch.abs(color_map - torch.roll(color_map, shifts=(dy, dx), dims=(2, 3))).sum(dim=1)
                # shape = (B,H,W)
                weighted_dist = shifted_dist + weighted_grad_map[:, 0, :, :] + color_diff * color_weight
                # shape = (B,H,W)

                # Hard STE for "update where new distance is smaller"
                update_mask = HardUpdateMask.apply(weighted_dist, dist_map)  # shape = (B,H,W)
                
                # Actually update dist_map and mask using the boolean
                dist_map = torch.where(update_mask.bool(), weighted_dist, dist_map)
                mask = torch.where(update_mask.bool(), shifted_mask, mask)
        
        return mask
        
    def forward(self, x):
        B, C_in, H, W = x.shape
        
        if C_in == 3:
            grayscale_image = self.convert_to_greyscale(x)
        else:
            grayscale_image = x
        
        # 1) Learnable gradient map
        grad_map = self.compute_gradient_map(grayscale_image)  # shape (B,1,H,W)

        # 2) Place centroids
        centroids = self.place_centroids_on_grid(B)

        # 3) Snap centroids to local minima (still purely discrete)
        centroids = self.find_nearest_minima(centroids, grad_map)

        # 4) Hard Voronoi with STE to allow some gradient
        mask = self.distance_weighted_propagation(centroids, grad_map, x)

        return grad_map, centroids, mask

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

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)

import random
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

NUM_CLUSTERS = 196

model = VoronoiPropagation(NUM_CLUSTERS, device='cuda').to('cuda')

optimizer = AdamW(
        model.parameters(),
        lr=1e-3
)

for group in optimizer.param_groups:
    for p in group["params"]:
        print(p.shape, p.requires_grad)


loss_fn = ExplainedVariationLoss(NUM_CLUSTERS)


max_epochs = 10


import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = mpl.colors.ListedColormap(torch.rand(256**2,3).numpy())

import matplotlib.pyplot as plt

def plot_sample_and_segments(source, mask, index=0, unnormalize=True):
    """
    source: (B, 3, H, W) or (B, 1, H, W)  -- batch of images
    mask:   (B, H, W)                    -- batch of segment masks (integer cluster IDs)
    index:  which sample in the batch to visualize
    unnormalize: if True, we'll attempt to invert typical ImageNet normalization for display
    """
    image = source[index].detach().cpu()  # shape (C,H,W)
    seg   = mask[index].detach().cpu()    # shape (H,W), integer cluster IDs

    if unnormalize and image.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)
        image = image * std + mean 
    # Convert image to (H,W,C) for plotting
    image_np = image.permute(1,2,0).numpy()
    if image_np.shape[2] == 1:
        image_np = image_np[:,:,0]

    seg_np = seg.numpy()  # shape (H,W)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    if image_np.ndim == 2:
        plt.imshow(image_np, cmap='gray', vmin=0, vmax=1)
    else:
        image_np = image_np.clip(0,1)  # in case any pixel is out of [0,1]
        plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(seg_np, cmap=cmap)
    plt.title("Segments")
    plt.axis('off')

    plt.tight_layout()
    plt.show()




for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    for batch_idx, source in enumerate(train_loader):
        source = source.to('cuda')
        optimizer.zero_grad()
        
        grad_map, centroids, mask = model(source)

        # Convert hard mask (B,H,W) -> one-hot (B,C,H,W) for the loss
        mask_one_hot = F.one_hot(mask.long(), num_classes=NUM_CLUSTERS)
        mask_one_hot = mask_one_hot.permute(0,3,1,2).float()  # shape (B,C,H,W)
        
        loss, ev_per_sample = loss_fn(source, mask_one_hot)
        
        loss.requires_grad = True
        
        loss.backward()
        
        print("Grad for edge_filter.weight:", model.edge_filter.weight.grad)
        print("Grad norm:", model.edge_filter.weight.grad.norm())
                
        optimizer.step()
        
        # For debug:
        for name, param in model.named_parameters():
            if 'edge_filter' in name:
                print(f"{name}: weight norm = {param.data.norm().item():.6f}, "
                    f"grad norm = {param.grad.norm().item():.6f}")
        
        ev_batch_mean = ev_per_sample.mean().item()
        print(f"Batch {batch_idx+1}: Loss = {loss.item():.4f}, EV (batch) = {ev_batch_mean:.4f}")
        print("Per-sample EV:", ev_per_sample.tolist())
        
        if batch_idx % 2 == 0:
            plot_sample_and_segments(source, mask, index=0, unnormalize=True)
        
        