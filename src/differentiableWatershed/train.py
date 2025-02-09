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

import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
import matplotlib.pyplot as plt
import random

# -------------------------------
# BSDS500 Dataset
# -------------------------------
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
        
        # Ground truth segmentation is loaded but not used in reconstruction.
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
    
    
class VoronoiPropagation(nn.Module):
    def __init__(self, num_clusters=196, height=224, width=224, device='cpu'):
        """
        Args:
            num_clusters (int): Number of clusters (centroids) to initialize.
            height (int): Height of the input image.
            width (int): Width of the input image.
            device (str): Device to run the model ('cpu' or 'cuda').
        """
        super(VoronoiPropagation, self).__init__()
        
        self.C = num_clusters
        self.H = height
        self.W = width
        self.device = torch.device(device)
        
        # Set bandwidth / sigma for kernel
        self.std = self.C / (self.H * self.W)**0.5

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
                
                # Iterate over all minima to find an unoccupied one
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
                    # If all minima are occupied, keep the original position
                    updated_centroids_batch.append([y.item(), x.item()])
            
            updated_centroids.append(torch.tensor(updated_centroids_batch, device=self.device))
        
        return torch.stack(updated_centroids, dim=0)

    def distance_weighted_propagation(self, centroids, grad_map, color_map, num_iters=50, gradient_weight=10.0, color_weight=10.0, edge_exponent=4.0): # gradient weight, color weight and edge exponent are all tuneable parameters 
        """
        Perform Voronoi-like propagation from centroids, guided by both the gradient map and color similarity.
        
        Args:
            centroids (Tensor): Initial centroid positions.
            grad_map (Tensor): Gradient magnitude map.
            color_map (Tensor): Input image for color similarity.
            num_iters (int): Number of iterations to perform propagation.
            gradient_weight (float): Weight for the gradient penalty.
            color_weight (float): Weight for the color similarity penalty.
            edge_exponent (float): Exponent to amplify edge gradients.
        
        Returns:
            Tensor: Final segmentation mask.
        """
        B, _, H, W = grad_map.shape
        mask = torch.full((B, H, W), fill_value=-1, device=grad_map.device)  # Label mask
        dist_map = torch.full((B, H, W), fill_value=float('inf'), device=grad_map.device)  # Distance map
        
        for batch_idx in range(B):
            for idx, (cy, cx) in enumerate(centroids[batch_idx]):
                mask[batch_idx, int(cy), int(cx)] = idx
                dist_map[batch_idx, int(cy), int(cx)] = 0  # Distance from centroid is 0 initially
        
        # 4-connected neighbors (dy, dx)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Amplify the impact of the gradient map by multiplying it with a weight and applying a non-linear transformation
        weighted_grad_map = (grad_map ** edge_exponent) * gradient_weight

        # Perform propagation with both gradient penalties and color similarity
        for _ in range(num_iters):
            for dy, dx in directions:
                # Shift the distance map in each direction
                shifted_dist = torch.roll(dist_map, shifts=(dy, dx), dims=(1, 2))
                shifted_mask = torch.roll(mask, shifts=(dy, dx), dims=(1, 2))
                
                # Calculate color distance between current pixel and centroid it is being propagated from
                color_diff = torch.abs(color_map - torch.roll(color_map, shifts=(dy, dx), dims=(2, 3))).sum(dim=1)  # Sum over color channels

                # Add the gradient map value as a weighted penalty to the distance
                weighted_dist = shifted_dist + weighted_grad_map[:, 0, :, :] + color_diff * color_weight
                
                # Update the mask and distance map where the new combined distance is smaller
                update_mask = weighted_dist < dist_map
                dist_map[update_mask] = weighted_dist[update_mask]
                mask[update_mask] = shifted_mask[update_mask]
        
        return mask
        
    def forward(self, x, grad_map):
        B, C_in, H, W = x.shape
        
        grad_map = grad_map.unsqueeze(1)
        
        # Place centroids on a grid
        centroids = self.place_centroids_on_grid(B)
        
        # Move centroids to nearest local minima
        centroids = self.find_nearest_minima(centroids, grad_map)
        
        # Perform distance-weighted propagation with both gradient and color guidance
        mask = self.distance_weighted_propagation(centroids, grad_map, x)
        
        # return grad_map, centroids, mask, spixel_features
        return centroids, mask


# -------------------------------
# DifferentiableSuperpixelTokenizer with Reconstruction MLP
# -------------------------------
class DifferentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, max_segments, n_channels=3, sobel_init=True, embed_dim=768,
                 use_positional_embeddings=True, reconstruction=False, device='cuda'):
        """
        Args:
            max_segments (int): Maximum number of superpixel segments.
            n_channels (int): Number of input image channels.
            sobel_init (bool): Whether to initialize the Sobel edge detection filter.
            embed_dim (int): Embedding dimension for the token representations.
            use_positional_embeddings (bool): If True, add positional embeddings.
            reconstruction (bool): If True, add an MLP head to reconstruct the input RGB image.
            device (str): Device to run the model.
        """
        super().__init__()
        self.superpixel_tokenizer = VoronoiPropagation(max_segments, height=224, width=224, device=device)
        self.max_segments = max_segments
        self.embed_dim = embed_dim
        self.use_positional_embeddings = use_positional_embeddings
        self.reconstruction = reconstruction

        # CNN backbone to extract feature maps
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        
        # Sobel edge detection
        self.edge_detection_conv = nn.Conv2d(
            in_channels=1,
            out_channels=2,  # [grad_x, grad_y]
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        if sobel_init:
            # Define Sobel kernels for Gx and Gy
            Gx = torch.tensor([[-1.,  0.,  1.],
                                [-2.,  0.,  2.],
                                [-1.,  0.,  1.]])
            Gy = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]])
            # Stack Gx and Gy so that shape = (out_channels, in_channels, kernel_height, kernel_width)
            sobel_kernel = torch.stack([Gx, Gy])  # shape: (2, 3, 3)
            sobel_kernel = sobel_kernel.unsqueeze(1)  # shape: (2, 1, 3, 3)
            self.edge_detection_conv.weight = nn.Parameter(sobel_kernel)
        
        if self.use_positional_embeddings:
            self.positional_embedding = nn.Linear(2, embed_dim)

        # Fusion layer to combine [mean_emb | max_emb] into one embedding of size embed_dim
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)

        # Reconstruction head: an MLP similar to transformer MLPs.
        # This maps from embed_dim to n_channels with an intermediate hidden layer.
        if self.reconstruction:
            hidden_dim = embed_dim * 4  # Adjust multiplier as needed.
            self.reconstruction_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_channels)
            )

    def forward(self, img):
        """
        Args:
            img (Tensor): Input image tensor of shape [B, n_channels, H, W].
        Returns:
            If reconstruction is False:
                final_embeddings (Tensor): [B, max_segments, embed_dim]
            If reconstruction is True:
                A tuple (final_embeddings, reconstructed_img), where:
                    - final_embeddings: [B, max_segments, embed_dim]
                    - reconstructed_img: [B, n_channels, H, W]
        """
        batch_size, n_channels, height, width = img.shape

        # 1) Extract features (backbone)
        features = self.cnn(img)  # [B, embed_dim, H_out, W_out]
        B, C, Hf, Wf = features.shape

        # 2) Compute gradient map using Sobel.
        gray_img = torch.mean(img, dim=1, keepdim=True)  # [B, 1, H, W]
        edges = self.edge_detection_conv(gray_img)  # [B, 2, H, W]
        grad_x = edges[:, 0, :, :]  # [B, H, W]
        grad_y = edges[:, 1, :, :]  # [B, H, W]
        gradient_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # [B, H, W]
    
        # 3) Tokenize into superpixels.
        centroid_coords, segments = self.superpixel_tokenizer(img, gradient_map)
        # segments: [B, H, W]; centroid_coords: [B, n_centroids, 2]

        # 4) Compute similarity measure: S(∇x) = 1 - grad_map
        similarity = 1.0 - gradient_map
        similarity = torch.clamp(similarity, 0.0, 1.0)  # [B, H, W]
        
        segments_flat = segments.view(B, -1)      # [B, H*W]
        similarity_flat = similarity.view(B, -1)  # [B, H*W]

        n_K = torch.zeros((B, self.max_segments), device=img.device).scatter_add(
            dim=1,
            index=segments_flat,
            src=torch.ones_like(similarity_flat)
        ).clamp(min=1)
        similarity_sum = torch.zeros((B, self.max_segments), device=img.device).scatter_add(
            dim=1,
            index=segments_flat,
            src=similarity_flat
        )
        W_k = similarity_sum / n_K  # [B, max_segments]

        # Aggregation of CNN features into superpixel embeddings:
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*Hf*Wf, C]
        segments_flat = segments.view(-1)  # [B*Hf*Wf]
        batch_indices = torch.arange(B, device=img.device).unsqueeze(1).expand(B, Hf * Wf).reshape(-1)
        unique_segment_ids = batch_indices * self.max_segments + segments_flat  # [B*Hf*Wf]
        dim_size = B * self.max_segments

        embeddings_mean = scatter_mean(features_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        embeddings_mean = embeddings_mean.view(B, self.max_segments, C)
        embeddings_max, _ = scatter_max(features_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        embeddings_max = embeddings_max.view(B, self.max_segments, C)
        embeddings_concat = torch.cat([embeddings_mean, embeddings_max], dim=-1)  # [B, max_segments, 2*C]
        embeddings_fused = self.fusion(embeddings_concat)  # [B, max_segments, embed_dim]
        weighted_embeddings = embeddings_fused * W_k.unsqueeze(-1)  # [B, max_segments, embed_dim]

        if self.use_positional_embeddings:
            centroids_normalized = centroid_coords.clone().float()
            centroids_normalized[:, :, 0] /= float(width)   # x / width
            centroids_normalized[:, :, 1] /= float(height)    # y / height
            pos_embeddings = self.positional_embedding(centroids_normalized.to(img.device))  # [B, n_centroids, embed_dim]
            n_centroids = pos_embeddings.shape[1]
            pos_embeddings_padded = torch.zeros(B, self.max_segments, self.embed_dim, device=img.device)
            if n_centroids > self.max_segments:
                pos_embeddings_padded = pos_embeddings[:, :self.max_segments, :]
            else:
                pos_embeddings_padded[:, :n_centroids, :] = pos_embeddings
            final_embeddings = weighted_embeddings + pos_embeddings_padded
        else:
            final_embeddings = weighted_embeddings

        if self.reconstruction:
            # superpixel_recon has shape [B, max_segments, n_channels]
            superpixel_recon = self.reconstruction_head(final_embeddings)
 
            # segments: [B, H, W] with integer values in [0, max_segments-1]

            #   Expand from [B, max_segments, n_channels] -> [B, H, max_segments, n_channels]
            superpixel_recon_exp = superpixel_recon.unsqueeze(1).expand(-1, height, -1, -1)
            
            # segments is [B, H, W]. Unsqueeze to [B, H, W, 1] and then expand to [B, H, W, n_channels]
            index = segments.unsqueeze(-1).expand(-1, -1, -1, n_channels)
            
            reconstructed_img = torch.gather(superpixel_recon_exp, dim=2, index=index)
            
            reconstructed_img = reconstructed_img.permute(0, 3, 1, 2)
            
            return final_embeddings, reconstructed_img, segments

        return final_embeddings

def explained_variance_batch(image_batch, superpixel_labels_batch):
    batch_size, num_channels, height, width = image_batch.shape
    explained_variance_scores = []

    for i in range(batch_size):
        image = image_batch[i]  # Shape: (C, H, W)
        superpixel_labels = superpixel_labels_batch[i]  # Shape: (H, W)

        # Ensure superpixel_labels is in shape (H, W)
        superpixel_labels = superpixel_labels.squeeze().to(image.device)

        # Flatten image and labels for computation
        image_flat = image.view(num_channels, height * width)
        labels_flat = superpixel_labels.view(height * width)

        # Compute total variance of the image across all channels
        total_variance = image_flat.var(dim=1, unbiased=False).mean().item()

        # Compute within-superpixel variance
        unique_labels = superpixel_labels.unique()
        num_superpixels = unique_labels.size(0)

        pixel_sums = torch.zeros((num_superpixels, num_channels), device=image.device)
        pixel_squares = torch.zeros((num_superpixels, num_channels), device=image.device)
        pixel_counts = torch.zeros(num_superpixels, device=image.device)

        for j, label in enumerate(unique_labels):
            mask = (labels_flat == label)
            if mask.sum() == 0:
                continue
            pixel_sums[j] = image_flat[:, mask].sum(dim=1)
            pixel_squares[j] = (image_flat[:, mask] ** 2).sum(dim=1)
            pixel_counts[j] = mask.sum()

        pixel_means = pixel_sums / pixel_counts.unsqueeze(1)
        pixel_variances = (pixel_squares / pixel_counts.unsqueeze(1)) - (pixel_means ** 2)
        within_variance = pixel_variances.mean().item()

        # Compute explained variance
        explained_variance = 1 - (within_variance / total_variance)
        explained_variance_scores.append(explained_variance)

    return explained_variance_scores

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
dataset_train = BSDS500Dataset(root_dir=r'D:\Data\BSDS500\data', split='train', transform=transform)
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)

# Training setup
NUM_CLUSTERS = 196
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DifferentiableSuperpixelTokenizer(
    max_segments=NUM_CLUSTERS,
    n_channels=3,
    use_positional_embeddings=False,
    reconstruction=True,
    embed_dim=192,
    device=device
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
max_epochs = 10

# Training loop
for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    model.train()
    running_loss = 0.0
    ev_accum = [] 
    
    for batch_idx, source in enumerate(train_loader):
        source = source.to(device)
        optimizer.zero_grad()
        
        final_embeddings, reconstructed_img, segments = model(source)
        
        loss = loss_fn(reconstructed_img, source)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        ev_scores = explained_variance_batch(source, segments)
        batch_ev = np.mean(ev_scores)
        ev_accum.append(batch_ev)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Batch Explained Variance: {batch_ev:.4f}")
    
    avg_loss = running_loss / len(train_loader)
    avg_ev = np.mean(ev_accum)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch+1} Average Explained Variance: {avg_ev:.4f}")
    
    sample_idx = random.randint(0, source.size(0) - 1)
    source_img = source[sample_idx].detach().cpu()
    recon_img = reconstructed_img[sample_idx].detach().cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    source_img = source_img * std + mean
    recon_img = recon_img * std + mean
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(source_img.permute(1,2,0).clamp(0,1))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(recon_img.permute(1,2,0).clamp(0,1))
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()