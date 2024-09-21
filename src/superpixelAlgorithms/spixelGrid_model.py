import torch
import torch.nn as nn


class LearnableGridSuperpixel(nn.Module):
    def __init__(self, height, width, grid_size):
        """
        Initializes the model with a regular grid that avoids cutting off edge pixels.
        
        Args:
            height: The height of the input image.
            width: The width of the input image.
            grid_size: Tuple (grid_h, grid_w) indicating the number of grid cells in each dimension.
        """
        super(LearnableGridSuperpixel, self).__init__()
        self.height = height
        self.width = width
        self.grid_h, self.grid_w = grid_size
        
        # Adjust grid lines by a small offset to avoid cutting pixels on edges
        h_interval = height / grid_size[0]
        w_interval = width / grid_size[1]
        
        # Center the grid lines by placing them slightly inward
        self.horizontal_lines = nn.Parameter(torch.linspace(h_interval / 2, height - h_interval / 2, self.grid_h), requires_grad=True)
        self.vertical_lines = nn.Parameter(torch.linspace(w_interval / 2, width - w_interval / 2, self.grid_w), requires_grad=True)
        
    def forward(self, x):
        """
        Forward pass that segments the image based on the learned grid lines.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width).
            
        Returns:
            segmentation_labels: Segmentation labels of the input image.
        """
        batch_size, channels, height, width = x.shape
        
        # Ensure grid lines are sorted and adjusted to not cut edges
        self.horizontal_lines.data = torch.sort(self.horizontal_lines)[0]
        self.vertical_lines.data = torch.sort(self.vertical_lines)[0]

        # Perform hard segmentation using learned grid lines
        segmentation_labels = self.create_hard_superpixel_mask(height, width)
        
        return None, None, segmentation_labels
    
    def create_hard_superpixel_mask(self, height, width):
        """
        Create a hard mask where each pixel is assigned to a superpixel based on proximity to grid lines.
        
        Args:
            height: The height of the image.
            width: The width of the image.
        
        Returns:
            A segmentation mask with shape (height, width) where each pixel has a label indicating the superpixel it belongs to.
        """
        y = torch.linspace(0, height - 1, height).to("cuda")
        x = torch.linspace(0, width - 1, width).to("cuda")
        
        yy, xx = torch.meshgrid(y, x)
        
        # Compute distances to the grid lines and assign to nearest superpixel region
        # Find the nearest horizontal and vertical grid line for each pixel
        h_distances = torch.abs(yy.unsqueeze(-1) - self.horizontal_lines).to("cuda")  # (height, width, grid_h)
        v_distances = torch.abs(xx.unsqueeze(-1) - self.vertical_lines).to("cuda")    # (height, width, grid_w)

        # Get the index of the nearest horizontal and vertical grid line
        nearest_h_line = torch.argmin(h_distances, dim=-1)  # (height, width)
        nearest_v_line = torch.argmin(v_distances, dim=-1)  # (height, width)

        # Combine the nearest horizontal and vertical line indices to create unique superpixel labels
        segmentation_labels = nearest_h_line * self.grid_w + nearest_v_line
        
        return segmentation_labels