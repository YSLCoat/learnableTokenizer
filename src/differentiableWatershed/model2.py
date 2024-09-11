import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

class VoronoiPropagation(nn.Module):
    def __init__(self, num_clusters=50, height=224, width=224, learnable_centroids=False):
        super(VoronoiPropagation, self).__init__()
        self.num_clusters = num_clusters
        self.height = height
        self.width = width
        self.learnable_centroids = learnable_centroids
        
        if learnable_centroids:
            self.sy = nn.Parameter(torch.rand(1, num_clusters))
            self.sx = nn.Parameter(torch.rand(1, num_clusters))  
        else:
            # Sobel filters for gradient computation
            self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            sobel_x_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sobel_y_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.sobel_x.weight = nn.Parameter(sobel_x_filter, requires_grad=False)
            self.sobel_y.weight = nn.Parameter(sobel_y_filter, requires_grad=False)
        
        # Learnable bandwidth/sigma for the kernel
        self.std = nn.Parameter(torch.tensor(num_clusters / (height * width) ** 0.5))
    
    def forward(self, combined_features):
        batch_size = combined_features.shape[0]
        
        # Assume combined_features includes both features and edges, so we'll work with both
        edges = combined_features[:, -1:]  # Assuming last channel is edges
        features = combined_features[:, :-1]  # Rest are features

        if self.learnable_centroids:
            # Use fixed learnable centroids
            sy = self.sy
            sx = self.sx
        else:
            # Compute gradients using Sobel filters
            Gx = self.sobel_x(edges)  # Gradient in x direction
            Gy = self.sobel_y(edges)  # Gradient in y direction
            gradient_magnitude = torch.sqrt(Gx ** 2 + Gy ** 2)
            
            # Invert the gradient magnitude to highlight low-gradient regions
            weight = 0.5  # This value can be fine-tuned
            inverted_gradient = 1.0 - (weight * gradient_magnitude)
            
            # Downsample the gradient to approximate centroid positions
            downsampled = F.adaptive_avg_pool2d(inverted_gradient, (self.num_clusters, self.num_clusters))
            
            # Flatten downsampled tensor and use topk to find the top `num_clusters` values for each image
            downsampled_flat = downsampled.view(batch_size, -1)
            topk_vals, topk_indices = torch.topk(downsampled_flat, self.num_clusters, dim=1)
            
            # Convert the 1D indices to 2D coordinates (row, col) in the downsampled space
            sy = topk_indices // self.num_clusters
            sx = topk_indices % self.num_clusters
            
            # Normalize the coordinates (sy, sx) to the [0, 1] range
            sy = sy.float() / (self.num_clusters)
            sx = sx.float() / (self.num_clusters)
        
        # Get the total number of pixels in the batch
        N = batch_size * self.height * self.width
        
        # Create shape tensor
        shape = torch.tensor([batch_size, self.height, self.width, 1], device=combined_features.device)[:, None]
        
        # Calculate pixel coordinates
        coefs = shape[1:].flipud().cumprod(dim=0).flipud()
        byx = torch.div(torch.arange(N, device=combined_features.device)[None], coefs, rounding_mode='trunc') % shape[:-1]

        # Normalize y, x coordinates
        y = byx[1] / self.height  # Shape: [N]
        x = byx[2] / self.width   # Shape: [N]

        # Reshape y and x to include the batch dimension for broadcasting
        y = y.view(batch_size, self.height * self.width, 1)  # Shape: [B, HW, 1]
        x = x.view(batch_size, self.height * self.width, 1)  # Shape: [B, HW, 1]
        
        # Compute L2 distance using the centroids (this time with both spatial and feature distances)
        feature_centroids = torch.mean(features, dim=[2, 3])  # Average features per superpixel

        # Combine spatial and feature distances (you can adjust weight here)
        spatial_distance = gauss2d(y - sy.unsqueeze(1), x - sx.unsqueeze(1), self.std)
        feature_distance = F.pairwise_distance(features.view(batch_size, -1), feature_centroids.unsqueeze(2).repeat(1, 1, self.height * self.width).view(batch_size, -1))

        # Combine both distance metrics
        combined_distance = spatial_distance + feature_distance.view(batch_size, self.num_clusters, self.height, self.width)

        # Softmax to get soft Voronoi regions
        markers = F.softmax(combined_distance, dim=1).view(batch_size, self.num_clusters, self.height, self.width)
        
        return markers

def gauss1d(x, std): return x.div(std).pow_(2).neg_().exp_()
def gauss2d(x, y, std): return (gauss1d(x, std) + gauss1d(y, std)) / 2


class UNetWithEdgeDetection(nn.Module):
    def __init__(self):
        super(UNetWithEdgeDetection, self).__init__()
        self.unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=64)  # Output: feature map
        self.learnable_edge_filter = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Learnable edge detection
    
    def forward(self, x):
        features = self.unet(x)
        edges = self.learnable_edge_filter(features)
        return features, edges


class DifferentiableWatershedWithVoronoi(nn.Module):
    def __init__(self, num_markers=3, height=224, width=224):
        super(DifferentiableWatershedWithVoronoi, self).__init__()
        
        # Voronoi propagation module
        self.voronoi_propagation = VoronoiPropagation(num_clusters=num_markers, height=height, width=width)
        
        # U-Net with learnable edge detection
        self.unet_with_edge_detection = UNetWithEdgeDetection()

    def forward(self, image):
        # Pass the image through U-Net to get features and edges
        features, edges = self.unet_with_edge_detection(image)
        
        # Concatenate features with edges for richer input to Voronoi propagation
        combined_features = torch.cat((features, edges), dim=1)
        
        # Use Voronoi propagation on the combined features to assign labels
        labels = self.voronoi_propagation(combined_features)
        
        return labels


