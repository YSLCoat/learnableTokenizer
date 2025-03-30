
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from timm.models._manipulate import checkpoint_seq
from torch_scatter import scatter_mean, scatter_max

class DifferentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, max_segments, n_channels=3, sobel_init=True, embed_dim=768,
                 use_positional_embeddings=True, reconstruction=False, device='cuda', superpixel_algorithm=None):
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
        # Select the superpixel algorithm based on the input string.
        algorithm = 'voronoi_propagation' if superpixel_algorithm is None else superpixel_algorithm.lower()
        if algorithm == 'voronoi_propagation':
            self.superpixel_algorithm = VoronoiPropagation(max_segments, height=224, width=224, device=device)
        elif algorithm == 'boundary_path_finder':
            self.superpixel_algorithm = BoundaryPathFinder(sqrt(max_segments), sqrt(max_segments), height=224, width=224, device=device)
        elif algorithm == 'slic_segmentation':
            self.superpixel_algorithm = SLICSegmentation(max_segments, height=224, width=224, device=device)
        else:
            raise ValueError(f"Unknown superpixel algorithm: {superpixel_algorithm}")
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
        centroid_coords, segments = self.superpixel_algorithm(img, gradient_map)
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
            
            return final_embeddings, reconstructed_img, segments, gradient_map

        return final_embeddings, gradient_map, segments

        

class DifferentiableSuperpixelTokenizerViT(nn.Module):
    def __init__(self, model_name, max_segments, num_classes, num_channels, superpixel_algorithm='voronoi_propagation', pretrained=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
        self.embed_dim = self.vit.embed_dim

        # Replace the patch embedding with the superpixel tokenizer
        self.vit.patch_embed = DifferentiableSuperpixelTokenizer(
            max_segments=max_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim,
            superpixel_algorithm=superpixel_algorithm,
        )

        # Remove positional embeddings from the ViT
        self.vit.pos_embed = None  # Positional embeddings are added in the tokenizer
        self.vit.num_tokens = max_segments + 1  # Update the number of tokens

        # Optionally, add positional embedding for the CLS token
        self.cls_positional_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings, gradient_map, segments = self.vit.patch_embed(x)  # [B, max_segments, embed_dim]
        b, n, d = embeddings.shape

        cls_tokens = self.vit.cls_token.expand(b, -1, -1)  # [B, 1, D]
        cls_tokens = cls_tokens + self.cls_positional_embedding  # Add positional embedding to CLS token

        x = torch.cat((cls_tokens, embeddings), dim=1)  # [B, n+1, D]

        x = self.vit.pos_drop(x)
        x = self.vit.norm_pre(x)

        if self.vit.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.vit.blocks, x)
        else:
            x = self.vit.blocks(x)

        x = self.vit.norm(x)
        return x, gradient_map, segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gradient_map, segments = self.forward_features(x)
        x = self.vit.forward_head(x, pre_logits=False)
        return x, gradient_map, segments
    
    
def test_tokenizer():
    # Define parameters
    max_segments = 196
    n_channels = 3
    embed_dim = 768
    use_pos_emb = True  # Change to False to test without positional embeddings
    
    # Instantiate the tokenizer
    tokenizer = DifferentiableSuperpixelTokenizer(
        max_segments=max_segments, 
        n_channels=n_channels, 
        sobel_init=True, 
        embed_dim=embed_dim,
        use_positional_embeddings=use_pos_emb
    )
    
    # Create a dummy image tensor: batch_size = 2, 3 channels, 224 x 224
    B, H, W = 2, 224, 224
    dummy_img = torch.randn(B, n_channels, H, W)
    
    # Forward pass
    output_embeddings = tokenizer(dummy_img)
    print("Output shape:", output_embeddings.shape)
    
def test_vit():
    # Define parameters for the ViT model
    model_name = "vit_base_patch16_224"  # Example model name from timm
    max_segments = 196
    num_classes = 10
    num_channels = 3

    # Instantiate the Vision Transformer with our differentiable tokenizer as patch_embed
    model = DifferentiableSuperpixelTokenizerViT(
        model_name=model_name,
        max_segments=max_segments,
        num_classes=num_classes,
        num_channels=num_channels,
        pretrained=False  # Change to True if you want to load pretrained weights
    )

    # Create a dummy image tensor: batch_size = 2, 3 channels, 224 x 224
    B, H, W = 2, 224, 224
    dummy_img = torch.randn(B, num_channels, H, W)

    # Forward pass through the ViT model
    output, gradient_map, segments = model(dummy_img)
    print("ViT output shape:", output.shape)

if __name__ == "__main__":
    test_tokenizer()
    test_vit()


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
                # color_diff = torch.abs(color_map - torch.roll(color_map, shifts=(dy, dx), dims=(2, 3))).sum(dim=1)  # Sum over color channels

                # Add the gradient map value as a weighted penalty to the distance
                weighted_dist = shifted_dist + weighted_grad_map[:, 0, :, :] #+ color_diff * color_weight
                
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

    
class BoundaryPathFinder(nn.Module):
    def __init__(self, num_segments_row=8, num_segments_col=8, height=224, width=224, device='cpu'):
        super(BoundaryPathFinder, self).__init__()
        
        self.num_segments_row = num_segments_row
        self.num_segments_col = num_segments_col
        self.H = height
        self.W = width
        self.device = device
        
        self.convert_to_grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
        
        
        # Move offsets for dynamic programming
        self.move_offsets = torch.tensor([-1, 0, 1], device=device)
    
    def initialize_grid(self, batch_size):
        # Create grid labels
        rows = torch.arange(self.H, device=self.device).unsqueeze(1)
        cols = torch.arange(self.W, device=self.device).unsqueeze(0)

        row_labels = rows // (self.H // self.num_segments_row)
        col_labels = cols // (self.W // self.num_segments_col)

        labels = (row_labels * self.num_segments_col + col_labels).to(torch.int32)
        labels = labels.expand(batch_size, -1, -1)  # Shape: (B, H, W)

        return labels
    
    def adjust_boundaries(self, grad_map, segmentation_mask, band_width=5):
        """
        Adjust boundary lines to align with the highest gradients while keeping the number of segments constant.
        """
        B, H, W = segmentation_mask.shape
        device = grad_map.device

        # Prepare indices
        y_indices = torch.arange(H, device=device)
        x_indices = torch.arange(W, device=device)

        # Initialize boundary masks
        boundary_masks_vertical = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        boundary_masks_horizontal = torch.zeros((B, H, W), dtype=torch.bool, device=device)

        # Vertical boundaries
        x_inits = torch.tensor([i * (W // int(self.num_segments_col)) for i in range(1, int(self.num_segments_col))], device=device).clamp(0, W - 1)
        num_vertical_paths = x_inits.size(0)
        for b in range(B):
            grad_map_b = grad_map[b, 0]  # Shape: (H, W)
            vertical_paths = self.find_optimal_vertical_paths(grad_map_b, x_inits, band_width)  # Shape: (num_vertical_paths, H)
            # Mark vertical boundaries
            for i in range(num_vertical_paths):
                boundary_masks_vertical[b, y_indices, vertical_paths[i]] = True

        # Horizontal boundaries
        y_inits = torch.tensor([i * (H // int(self.num_segments_row)) for i in range(1, int(self.num_segments_row))], device=device).clamp(0, H - 1)
        num_horizontal_paths = y_inits.size(0)
        for b in range(B):
            grad_map_b = grad_map[b, 0]  # Shape: (H, W)
            horizontal_paths = self.find_optimal_horizontal_paths(grad_map_b, y_inits, band_width)  # Shape: (num_horizontal_paths, W)
            # Mark horizontal boundaries
            for i in range(num_horizontal_paths):
                boundary_masks_horizontal[b, horizontal_paths[i], x_indices] = True

        # Compute vertical labels
        vertical_boundaries_int = boundary_masks_vertical.to(torch.int32)
        vertical_labels = torch.cumsum(vertical_boundaries_int, dim=2)

        # Compute horizontal labels
        horizontal_boundaries_int = boundary_masks_horizontal.to(torch.int32)
        horizontal_labels = torch.cumsum(horizontal_boundaries_int, dim=1)

        # Compute final region labels
        num_vertical_segments = self.num_segments_col
        num_horizontal_segments = self.num_segments_row

        new_segmentation_masks = vertical_labels + num_vertical_segments * horizontal_labels

        return new_segmentation_masks  # Shape: (B, H, W)

    def find_optimal_vertical_paths(self, grad_map, x_inits, band_width):
        """
        Find the optimal vertical paths around the initial x positions using dynamic programming.
        """
        H, W = grad_map.shape
        device = grad_map.device
        num_paths = x_inits.size(0)

        # Define bands around x_inits
        x_offsets = torch.arange(-band_width, band_width + 1, device=device)
        x_indices = x_inits.unsqueeze(1) + x_offsets.unsqueeze(0)  # Shape: (num_paths, num_positions)
        x_indices = x_indices.clamp(0, W - 1).long()
        num_positions = x_indices.size(1)

        # Initialize cost and path matrices
        cost = torch.full((H, num_paths, num_positions), float('inf'), device=device)
        path = torch.zeros((H, num_paths, num_positions), dtype=torch.long, device=device)

        # First row
        grad_row = grad_map[0].unsqueeze(0).expand(num_paths, -1)  # Shape: (num_paths, W)
        cost[0] = -grad_row.gather(1, x_indices)  # Shape: (num_paths, num_positions)

        # Precompute position indices
        positions = torch.arange(num_positions, device=device).unsqueeze(0).expand(num_paths, -1)  # Shape: (num_paths, num_positions)

        # Dynamic programming
        for y in range(1, H):
            # Pad the previous cost for easy indexing
            padded_prev_cost = torch.cat([
                torch.full((num_paths, 1), float('inf'), device=device),
                cost[y - 1],
                torch.full((num_paths, 1), float('inf'), device=device)
            ], dim=1)  # Shape: (num_paths, num_positions + 2)

            # Indices for possible moves: left (-1), stay (0), right (+1)
            move_offsets = torch.tensor([-1, 0, 1], device=device)
            neighbor_indices = positions.unsqueeze(2) + move_offsets.view(1, 1, -1)  # Shape: (num_paths, num_positions, 3)
            neighbor_indices = neighbor_indices.clamp(0, num_positions - 1)

            # Adjust for padding
            neighbor_indices_padded = neighbor_indices + 1  # Adjust for the padding
            neighbor_indices_padded = neighbor_indices_padded.long()

            # Adjust dimensions of padded_prev_cost
            padded_prev_cost_expanded = padded_prev_cost.unsqueeze(1).expand(-1, num_positions, -1)

            # Gather costs for possible moves
            prev_costs = padded_prev_cost_expanded.gather(2, neighbor_indices_padded)  # Shape: (num_paths, num_positions, 3)

            # Find the minimum cost among the neighbors
            min_prev_costs, min_indices = prev_costs.min(dim=2)  # Shape: (num_paths, num_positions)

            # Update cost and path
            grad_row = grad_map[y].unsqueeze(0).expand(num_paths, -1)  # Shape: (num_paths, W)
            current_grad = -grad_row.gather(1, x_indices)
            cost[y] = min_prev_costs + current_grad  # Shape: (num_paths, num_positions)
            path[y] = neighbor_indices.gather(2, min_indices.unsqueeze(2)).squeeze(2)  # Shape: (num_paths, num_positions)

        # Backtracking to find the optimal paths
        idx = cost[-1].argmin(dim=1)  # Shape: (num_paths,)
        optimal_paths = []
        for y in reversed(range(H)):
            optimal_paths.append(x_indices[torch.arange(num_paths), idx])  # Shape: (num_paths,)
            idx = path[y, torch.arange(num_paths), idx]
        optimal_paths = torch.stack(optimal_paths[::-1], dim=1)  # Shape: (num_paths, H)
        return optimal_paths  # Shape: (num_paths, H)

    def find_optimal_horizontal_paths(self, grad_map, y_inits, band_width):
        """
        Find the optimal horizontal paths around the initial y positions using dynamic programming.
        """
        H, W = grad_map.shape
        device = grad_map.device
        num_paths = y_inits.size(0)

        # Define bands around y_inits
        y_offsets = torch.arange(-band_width, band_width + 1, device=device)
        y_indices = y_inits.unsqueeze(1) + y_offsets.unsqueeze(0)  # Shape: (num_paths, num_positions)
        y_indices = y_indices.clamp(0, H - 1).long()
        num_positions = y_indices.size(1)

        # Initialize cost and path matrices
        cost = torch.full((W, num_paths, num_positions), float('inf'), device=device)
        path = torch.zeros((W, num_paths, num_positions), dtype=torch.long, device=device)

        # First column
        grad_col = grad_map[:, 0].unsqueeze(0).expand(num_paths, -1)  # Shape: (num_paths, H)
        cost[0] = -grad_col.gather(1, y_indices)  # Shape: (num_paths, num_positions)

        # Precompute position indices
        positions = torch.arange(num_positions, device=device).unsqueeze(0).expand(num_paths, -1)  # Shape: (num_paths, num_positions)

        # Dynamic programming
        for x in range(1, W):
            # Pad the previous cost for easy indexing
            padded_prev_cost = torch.cat([
                torch.full((num_paths, 1), float('inf'), device=device),
                cost[x - 1],
                torch.full((num_paths, 1), float('inf'), device=device)
            ], dim=1)  # Shape: (num_paths, num_positions + 2)

            # Indices for possible moves: up (-1), stay (0), down (+1)
            move_offsets = torch.tensor([-1, 0, 1], device=device)
            neighbor_indices = positions.unsqueeze(2) + move_offsets.view(1, 1, -1)  # Shape: (num_paths, num_positions, 3)
            neighbor_indices = neighbor_indices.clamp(0, num_positions - 1)

            # Adjust for padding
            neighbor_indices_padded = neighbor_indices + 1  # Adjust for the padding
            neighbor_indices_padded = neighbor_indices_padded.long()

            # Adjust dimensions of padded_prev_cost
            padded_prev_cost_expanded = padded_prev_cost.unsqueeze(1).expand(-1, num_positions, -1)

            # Gather costs for possible moves
            prev_costs = padded_prev_cost_expanded.gather(2, neighbor_indices_padded)  # Shape: (num_paths, num_positions, 3)

            # Find the minimum cost among the neighbors
            min_prev_costs, min_indices = prev_costs.min(dim=2)  # Shape: (num_paths, num_positions)

            # Update cost and path
            grad_col = grad_map[:, x].unsqueeze(0).expand(num_paths, -1)  # Shape: (num_paths, H)
            current_grad = -grad_col.gather(1, y_indices)
            cost[x] = min_prev_costs + current_grad  # Shape: (num_paths, num_positions)
            path[x] = neighbor_indices.gather(2, min_indices.unsqueeze(2)).squeeze(2)  # Shape: (num_paths, num_positions)

        # Backtracking to find the optimal paths
        idx = cost[-1].argmin(dim=1)  # Shape: (num_paths,)
        optimal_paths = []
        for x in reversed(range(W)):
            optimal_paths.append(y_indices[torch.arange(num_paths), idx])  # Shape: (num_paths,)
            idx = path[x, torch.arange(num_paths), idx]
        optimal_paths = torch.stack(optimal_paths[::-1], dim=1)  # Shape: (num_paths, W)
        return optimal_paths  # Shape: (num_paths, W)

    
    def compute_centroids(self, seg_mask):
        """
        seg_mask: (B, H, W) with label values in [0, K-1], K = num_segments_row * num_segments_col
        Returns a list of length B, each entry is a (K, 2) tensor [y, x] for each superpixel's centroid.
        """
        B, H, W = seg_mask.shape
        K = int(self.num_segments_row * self.num_segments_col)

        # We will store the centroids for each batch in a list of shape (B, K, 2).
        centroids_list = []

        for b in range(B):
            mask_b = seg_mask[b]  # shape: (H, W)
            # We'll create placeholders for sums of y, sums of x, and counts for each label
            sums_y = torch.zeros(K, device=self.device, dtype=torch.float32)
            sums_x = torch.zeros(K, device=self.device, dtype=torch.float32)
            counts = torch.zeros(K, device=self.device, dtype=torch.float32)

            # We can do this in a vectorized manner:
            # 1) Flatten the mask
            flat_mask = mask_b.view(-1)  # shape: (H*W,)
            # 2) Prepare coordinates
            y_coords = torch.arange(H, device=self.device).unsqueeze(1).expand(H, W).reshape(-1)
            x_coords = torch.arange(W, device=self.device).unsqueeze(0).expand(H, W).reshape(-1)
            
            # 3) Add to sums
            sums_y.index_add_(0, flat_mask, y_coords.float())
            sums_x.index_add_(0, flat_mask, x_coords.float())
            counts.index_add_(0, flat_mask, torch.ones_like(y_coords, dtype=torch.float32))

            # Now compute centroids = sums / counts
            # Some superpixels might have zero count if a boundary ended up not being used,
            # but normally each label should appear at least once. 
            # We can handle the possibility of zero counts by division with clamp_min.
            counts = counts.clamp_min(1e-6)
            centroids_y = sums_y / counts
            centroids_x = sums_x / counts

            # (K, 2) -> each row: [y, x]
            centroids_b = torch.stack([centroids_y, centroids_x], dim=-1)
            centroids_list.append(centroids_b)

        # Convert list to a single tensor if desired: (B, K, 2)
        centroids_all = torch.stack(centroids_list, dim=0)
        return centroids_all

    def forward(self, x, grad_map):
        B, C, H, W = x.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Input image size must match initialized size: ({self.H}, {self.W})")

        # Optionally compute the gradient map if not provided
        # grad_map = self.compute_gradient_map(x)  # shape: (B, 1, H, W)
        # But here we assume grad_map is provided externally and has shape (B, H, W):
        grad_map = grad_map.unsqueeze(1)

        # 1) Initialize grid segmentation
        segmentation_mask = self.initialize_grid(B)  # shape: (B, H, W)
        
        # 2) Adjust boundaries
        new_segmentation_mask = self.adjust_boundaries(grad_map, segmentation_mask).long()
        
        # 3) Compute centroids for each superpixel
        centroids = self.compute_centroids(new_segmentation_mask)
        # centroids shape: (B, K, 2)

        # Return the number of segments if needed, or you can remove it
        return centroids, new_segmentation_mask

class SLICSegmentation(nn.Module):
    def __init__(self, num_clusters=196, height=224, width=224, device='cpu'):
        """
        Args:
            num_clusters (int): Number of clusters (centroids) to initialize.
            height (int): Height of the input image.
            width (int): Width of the input image.
            device (str): Device to run the model ('cpu' or 'cuda').
        """
        super(SLICSegmentation, self).__init__()
        
        self.C = num_clusters
        self.H = height
        self.W = width
        self.device = torch.device(device)
        
        # Set bandwidth / sigma for kernel
        self.std = self.C / (self.H * self.W)**0.5

    def place_centroids_on_grid(self, batch_size):
        """
        Places centroids (seeds) roughly evenly on a grid across the image.
        """
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
        Moves each centroid to the nearest local minimum of the gradient map
        within a specified neighborhood.  Avoids collisions if possible.
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

    def SLIC_vectorized(self, centroids, x, max_iter=50, m=10.0):
        """
        Perform SLIC-like clustering to generate superpixels in a fully vectorized way.

        Args:
            centroids (Tensor):  (B, C, 2) initial centroid positions (y, x).
            x         (Tensor):  (B, C_in, H, W) input image or feature map.
            max_iter    (int):   Number of SLIC iterations.
            m         (float):   Weighting factor for spatial distance vs. color distance.

        Returns:
            label_map (Tensor): (B, H, W) of integer cluster assignments for each pixel.
        """
        device = x.device
        B, C_in, H, W = x.shape
        C = centroids.shape[1]

        # S ~ approximate spacing between clusters
        S = math.sqrt(H * W / C)
        # We'll use squared distance to avoid expensive sqrt
        m_s_sq = (m / S) ** 2

        # 1) Round & clamp the initial centroid positions
        yc = torch.round(centroids[..., 0]).long().clamp(0, H - 1)  # (B, C)
        xc = torch.round(centroids[..., 1]).long().clamp(0, W - 1)  # (B, C)

        # 2) Initialize centroid colors by sampling from x at those positions
        #    shape = (B, C, C_in)
        centroid_colors = []
        for b_idx in range(B):
            # gather all cluster colors in this image
            colors_b = []
            for c_idx in range(C):
                yy = yc[b_idx, c_idx]
                xx = xc[b_idx, c_idx]
                colors_b.append(x[b_idx, :, yy, xx])  # shape (C_in,)
            centroid_colors.append(torch.stack(colors_b, dim=0))  # (C, C_in)
        centroid_colors = torch.stack(centroid_colors, dim=0)  # (B, C, C_in)

        # 3) Precompute a full coordinate grid for spatial distances
        #    shapes => Y, X: (H, W) => expand to (1,1,H,W) for broadcast
        Y = torch.arange(H, device=device).view(1, 1, H, 1)
        X = torch.arange(W, device=device).view(1, 1, 1, W)

        # 4) Iterative updates
        for _ in range(max_iter):
            # Expand centroid colors to broadcast: (B, C, C_in) -> (B, C, C_in, 1, 1)
            centroid_colors_exp = centroid_colors.unsqueeze(-1).unsqueeze(-1)  # (B, C, C_in, 1, 1)

            # Expand image for color distance: (B, C_in, H, W) -> (B, 1, C_in, H, W)
            x_exp = x.unsqueeze(1)  # (B, 1, C_in, H, W)

            # color_dist_sq => (B, C, H, W)
            color_dist_sq = (x_exp - centroid_colors_exp).pow(2).sum(dim=2)  # sum over C_in

            # For spatial_dist, we need each centroid's (yc, xc):
            # yc, xc => (B, C). We expand to => (B, C, 1, 1)
            # Then subtract from coordinate grid shape => (1, 1, H, W)
            yc_exp = yc.view(B, C, 1, 1)
            xc_exp = xc.view(B, C, 1, 1)

            spatial_dist_sq = (Y - yc_exp).pow(2) + (X - xc_exp).pow(2)  # (B, C, H, W)

            dist_sq = color_dist_sq + m_s_sq * spatial_dist_sq  # (B, C, H, W)

            # 5) Argmin over cluster dimension => label each pixel
            label_map = dist_sq.argmin(dim=1)  # shape (B, H, W)

            # 6) Update the centroid positions & colors in a fully vectorized manner
            #    We'll do it by summing all pixel positions & colors assigned to each cluster,
            #    then dividing by the pixel count for each cluster.

            new_yc = torch.zeros_like(yc)  # (B, C)
            new_xc = torch.zeros_like(xc)  # (B, C)
            new_centroid_colors = torch.zeros_like(centroid_colors)  # (B, C, C_in)

            # We'll do the update per-batch to keep it simpler
            for b_idx in range(B):
                # Flatten the label_map for this image
                labels_b = label_map[b_idx].view(-1)  # shape (H*W,)
                # Flatten the color for this image: (C_in, H*W)
                color_b = x[b_idx].view(C_in, -1)
                # Coordinates for each pixel
                y_coords = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1)
                x_coords = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1)

                # We need to accumulate sums of y, sums of x, sums of colors, and counts
                # for each cluster ID in [0..C-1].
                # shape (C,) for sums of y, sums of x, counts
                sum_y = torch.zeros(C, device=device, dtype=torch.float)
                sum_x = torch.zeros(C, device=device, dtype=torch.float)
                count = torch.zeros(C, device=device, dtype=torch.float)

                # shape (C, C_in) for color sums
                sum_color = torch.zeros(C, C_in, device=device, dtype=torch.float)

                # We can use scatter_add_ or index_add_:
                sum_y.index_add_(0, labels_b, y_coords.float())
                sum_x.index_add_(0, labels_b, x_coords.float())
                count.index_add_(0, labels_b, torch.ones_like(labels_b, dtype=torch.float))

                # For color, we need to do it per channel
                # color_b: (C_in, H*W), labels_b: (H*W,)
                # We'll transpose color_b => (H*W, C_in) so we can index_add along dimension 0
                color_b_t = color_b.t()  # (H*W, C_in)
                # We do scatter-add or index_add for each channel
                # We'll do a loop over channels or create an index for each pixel–channel pair
                # but simpler is to do channel by channel in a loop:
                for c_in_idx in range(C_in):
                    sum_color_channel = sum_color[:, c_in_idx]  # (C,)
                    sum_color_channel.index_add_(0, labels_b, color_b_t[:, c_in_idx])
                    # store back
                    sum_color[:, c_in_idx] = sum_color_channel

                # Now compute new centroid positions/colors
                # Avoid dividing by zero for clusters that got no pixels
                # We'll keep old position/color if count == 0
                nonzero_mask = (count > 0)

                new_y = sum_y[nonzero_mask] / count[nonzero_mask]
                new_x = sum_x[nonzero_mask] / count[nonzero_mask]
                new_color = sum_color[nonzero_mask] / count[nonzero_mask].unsqueeze(-1)

                # Write them back into new_yc, etc.
                # cluster IDs for those non-empty clusters
                nonzero_ids = torch.nonzero(nonzero_mask, as_tuple=True)[0]
                new_yc[b_idx, nonzero_ids] = new_y.round().long().clamp(0, H - 1)
                new_xc[b_idx, nonzero_ids] = new_x.round().long().clamp(0, W - 1)
                new_centroid_colors[b_idx, nonzero_ids] = new_color

                # For empty clusters, keep old positions/colors
                # (Already in new_yc / new_xc / new_centroid_colors as zeros,
                #  so we can revert to old ones if you prefer)
                empty_ids = torch.nonzero(~nonzero_mask, as_tuple=True)[0]
                if len(empty_ids) > 0:
                    new_yc[b_idx, empty_ids] = yc[b_idx, empty_ids]
                    new_xc[b_idx, empty_ids] = xc[b_idx, empty_ids]
                    new_centroid_colors[b_idx, empty_ids] = centroid_colors[b_idx, empty_ids]

            # Save updated centroids
            yc = new_yc
            xc = new_xc
            centroid_colors = new_centroid_colors

        return label_map

    def forward(self, x, grad_map):
        """
        Forward pass:
          1) Initialize centroids on a grid.
          2) Optionally move them to local minima of grad_map.
          3) Run SLIC-like iterative assignment and update.
          4) Return final centroids and assignment mask.
        """
        B, C_in, H, W = x.shape
        
        # Make sure grad_map has shape (B, 1, H, W)
        if grad_map.ndim == 3:
            grad_map = grad_map.unsqueeze(1)
        
        # 1) Place centroids
        centroids = self.place_centroids_on_grid(B)
        
        # 2) Move centroids to nearest local minima in grad_map
        centroids = self.find_nearest_minima(centroids, grad_map)
        
        # 3) SLIC
        mask = self.SLIC_vectorized(centroids, x, max_iter=50, m=20.0)
        
        return centroids, mask
