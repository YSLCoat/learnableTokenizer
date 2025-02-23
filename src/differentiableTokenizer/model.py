import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import segmentation_models_pytorch as smp
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import segmentation_models_pytorch as smp


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

    
class BoundaryPathFinder(nn.Module):
    def __init__(self, num_segments_row=8, num_segments_col=8, height=224, width=224, device='cpu'):
        super(BoundaryPathFinder, self).__init__()
        
        self.num_segments_row = num_segments_row
        self.num_segments_col = num_segments_col
        self.H = height
        self.W = width
        self.device = device
        
        self.convert_to_grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
        
        # Sobel kernels
        self.sobel_x = torch.tensor([[[[-1, 0, 1], 
                                  [-2, 0, 2], 
                                  [-1, 0, 1]]]], device=device, dtype=torch.float32)
        self.sobel_y = torch.tensor([[[[-1, -2, -1], 
                                  [0, 0, 0], 
                                  [1, 2, 1]]]], device=device, dtype=torch.float32)
        
        # Move offsets for dynamic programming
        self.move_offsets = torch.tensor([-1, 0, 1], device=device)
    
    def compute_gradient_map(self, x):
        # x: (B, C, H, W)
        if x.shape[1] == 3:
            x = self.convert_to_grayscale(x)
        
        # Apply Sobel filters
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # Compute gradient magnitude
        grad_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad_map  # Shape: (B, 1, H, W)
    
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
        x_inits = torch.tensor([i * (W // self.num_segments_col) for i in range(1, self.num_segments_col)], device=device).clamp(0, W - 1)
        num_vertical_paths = x_inits.size(0)
        for b in range(B):
            grad_map_b = grad_map[b, 0]  # Shape: (H, W)
            vertical_paths = self.find_optimal_vertical_paths(grad_map_b, x_inits, band_width)  # Shape: (num_vertical_paths, H)
            # Mark vertical boundaries
            for i in range(num_vertical_paths):
                boundary_masks_vertical[b, y_indices, vertical_paths[i]] = True

        # Horizontal boundaries
        y_inits = torch.tensor([i * (H // self.num_segments_row) for i in range(1, self.num_segments_row)], device=device).clamp(0, H - 1)
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

    
    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Input image size must match initialized size: ({self.H}, {self.W})")

        # Compute gradient map
        grad_map = self.compute_gradient_map(x)  # Shape: (B, 1, H, W)

        # Initialize grid segmentation
        segmentation_mask = self.initialize_grid(B)  # Shape: (B, H, W)

        # Adjust boundaries
        new_segmentation_mask = self.adjust_boundaries(grad_map, segmentation_mask)

        return grad_map, segmentation_mask, new_segmentation_mask


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

    def SLIC(self, centroids, x, max_iter=2, m=10.0):
        """
        Perform a SLIC-like clustering to generate superpixels.
        
        Args:
            centroids (Tensor):  (B, C, 2) with (y, x) positions for each cluster.
            x         (Tensor):  (B, C_in, H, W) input image (e.g., RGB or any feature).
            max_iter    (int):   Number of SLIC iterations.
            m         (float):   Weighting factor for spatial distance vs. color distance.

        Returns:
            mask (Tensor): (B, H, W) of integer cluster assignments for each pixel.
        """
        B, C_in, H, W = x.shape
        
        # We'll define the superpixel spacing, S, as ~sqrt(area / num_clusters).
        S = int(math.sqrt(H * W / self.C) + 0.5)

        # Create label map (-1 initially) and distance map (inf initially).
        label_map = torch.full((B, H, W), -1, device=self.device, dtype=torch.long)
        distance_map = torch.full((B, H, W), float('inf'), device=self.device)
        
        # Extract integer centroid coordinates
        # (We round them so we can index directly into the image.)
        yc = centroids[..., 0].round().long()
        xc = centroids[..., 1].round().long()
        
        # Initialize "centroid colors" by sampling from x at those coords
        # shape: (B, C, C_in)
        # We'll do a gather for each batch to handle in a simple loop.
        centroid_colors = []
        for b_idx in range(B):
            colors_b = []
            for c_idx in range(self.C):
                yy = yc[b_idx, c_idx].clamp(0, H - 1)
                xx = xc[b_idx, c_idx].clamp(0, W - 1)
                colors_b.append(x[b_idx, :, yy, xx])
            centroid_colors.append(torch.stack(colors_b, dim=0))
        centroid_colors = torch.stack(centroid_colors, dim=0)  # (B, C, C_in)
        
        # Iterative update
        for _ in range(max_iter):
            # Step 1: Assign pixels to clusters within a local window of size ~ 2S
            label_map.fill_(-1)
            distance_map.fill_(float('inf'))
            
            for b_idx in range(B):
                for c_idx in range(self.C):
                    cy = yc[b_idx, c_idx].item()
                    cx = xc[b_idx, c_idx].item()
                    
                    # Neighborhood bounds
                    y_min = max(0, cy - S)
                    y_max = min(H, cy + S)
                    x_min = max(0, cx - S)
                    x_max = min(W, cx + S)
                    
                    # Extract color patch
                    color_patch = x[b_idx, :, y_min:y_max, x_min:x_max]
                    
                    # Distance in color space
                    c_col = centroid_colors[b_idx, c_idx].unsqueeze(-1).unsqueeze(-1)
                    # shape (C_in, 1, 1)
                    color_dist = (color_patch - c_col).pow(2).sum(dim=0).sqrt()
                    
                    # Distance in (y, x) / spatial space
                    yy, xx = torch.meshgrid(
                        torch.arange(y_min, y_max, device=self.device),
                        torch.arange(x_min, x_max, device=self.device),
                        indexing='ij'
                    )
                    spatial_dist = torch.sqrt((yy - cy)**2 + (xx - cx)**2)

                    # Combined distance (the 'm / S' factor for balancing color vs. spatial)
                    dist = color_dist + (m / S) * spatial_dist
                    
                    # Update where we get smaller distance
                    mask_region = distance_map[b_idx, y_min:y_max, x_min:x_max] > dist
                    distance_map[b_idx, y_min:y_max, x_min:x_max][mask_region] = dist[mask_region]
                    label_map[b_idx, y_min:y_max, x_min:x_max][mask_region] = c_idx
            
            # Step 2: Update cluster centroids (position + color)
            new_yc = yc.clone()
            new_xc = xc.clone()
            new_centroid_colors = centroid_colors.clone()
            
            for b_idx in range(B):
                for c_idx in range(self.C):
                    # All pixels assigned to cluster c_idx
                    mask_cluster = (label_map[b_idx] == c_idx)
                    if mask_cluster.any():
                        # Pixel coords
                        y_coords, x_coords = torch.nonzero(mask_cluster, as_tuple=True)
                        
                        # Update centroid position
                        cy_new = torch.mean(y_coords.float())
                        cx_new = torch.mean(x_coords.float())
                        
                        # Clamp + round
                        cy_new = cy_new.clamp(0, H - 1).round().long()
                        cx_new = cx_new.clamp(0, W - 1).round().long()
                        
                        new_yc[b_idx, c_idx] = cy_new
                        new_xc[b_idx, c_idx] = cx_new
                        
                        # Update centroid color
                        color_vals = x[b_idx, :, y_coords, x_coords]
                        new_centroid_colors[b_idx, c_idx] = color_vals.mean(dim=1)
                    # else: no pixels assigned -> keep old
        
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
        mask = self.SLIC(centroids, x, max_iter=2, m=10.0)
        
        return centroids, mask
