import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            sobel_x_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sobel_y_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.sobel_x.weight = nn.Parameter(sobel_x_filter, requires_grad=False)
            self.sobel_y.weight = nn.Parameter(sobel_y_filter, requires_grad=False)
        
        # Set bandwidth / sigma for kernel (learnable)
        self.std = nn.Parameter(torch.tensor(num_clusters / (height * width) ** 0.5))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.learnable_centroids:
            # Use fixed learnable centroids
            sy = self.sy
            sx = self.sx
        else:
            # Compute gradients using Sobel filters
            Gx = self.sobel_x(x)  # Gradient in x direction
            Gy = self.sobel_y(x)  # Gradient in y direction
            gradient_magnitude = torch.sqrt(Gx ** 2 + Gy ** 2)
            
            # Invert the gradient magnitude to highlight low-gradient regions
            inverted_gradient = 1.0 - gradient_magnitude
            
            # Downsample the gradient to approximate centroid positions
            downsampled = F.adaptive_avg_pool2d(inverted_gradient, (self.num_clusters, self.num_clusters))
            
            # Flatten downsampled tensor and use topk to find the top `num_clusters` values for each image
            downsampled_flat = downsampled.view(batch_size, -1)
            topk_vals, topk_indices = torch.topk(downsampled_flat, self.num_clusters, dim=1)
            
            # Convert the 1D indices to 2D coordinates (row, col) in the downsampled space
            sy = topk_indices // self.num_clusters
            sx = topk_indices % self.num_clusters
            
            # Normalize the coordinates (sy, sx) to the [0, 1] range
            sy = sy.float() / (self.num_clusters - 1)
            sx = sx.float() / (self.num_clusters - 1)
        
        # Get the total number of pixels in the batch
        N = batch_size * self.height * self.width
        
        # Create shape tensor
        shape = torch.tensor([batch_size, self.height, self.width, 1], device=device)[:, None]
        
        # Calculate pixel coordinates
        coefs = shape[1:].flipud().cumprod(dim=0).flipud()
        byx = torch.div(torch.arange(N, device=device)[None], coefs, rounding_mode='trunc') % shape[:-1]

        # Normalize y, x coordinates
        y = byx[1] / self.height  # Shape: [N]
        x = byx[2] / self.width   # Shape: [N]

        # Reshape y and x to include the batch dimension for broadcasting
        y = y.view(batch_size, self.height * self.width, 1)  # Shape: [B, HW, 1]
        x = x.view(batch_size, self.height * self.width, 1)  # Shape: [B, HW, 1]
        
        #assert 0, (y.shape, sy.shape)
        
        # Compute L2 distance using the centroids
        l2 = gauss2d(y - sy.unsqueeze(1), x - sx.unsqueeze(1), self.std)
        print(l2.shape)
        # Softmax to get soft Voronoi regions
       # markers = F.softmax(l2, dim=1).view(batch_size, self.num_clusters, self.height, self.width)
        
        return l2.view(-1,50)

def gauss1d(x, std): return x.div(std).pow_(2).neg_().exp_()
def gauss2d(x, y, std): return (gauss1d(x, std) + gauss1d(y, std)) / 2


class DifferentiableWatershedWithVoronoi(nn.Module):
    def __init__(self, num_markers=3, num_iterations=20, rnn_hidden_channels=51):
        super(DifferentiableWatershedWithVoronoi, self).__init__()

        self.num_markers = num_markers

        # Define Sobel kernels
        self.Kx = torch.tensor([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        self.Ky = torch.tensor([[ 1,  2,  1], 
                        [ 0,  0,  0], 
                        [-1, -2, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Convolutional layers for marker prediction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_markers, kernel_size=3, padding=1)  # Output channels = num_markers
        
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)  # No need to send to device, done in forward pass

        # Voronoi propagation module
        self.voronoi_propagation = VoronoiPropagation(num_clusters=num_markers, height=224, width=224)

        # Optional RNN for flooding approximation (could be used after Voronoi)
        # self.rnn_flooding = FloodingRNN(in_channels=51,  # Markers and gradient magnitude
        #                                 hidden_channels=rnn_hidden_channels, 
        #                                 out_channels=num_markers, 
        #                                 num_iterations=num_iterations)
        
    def forward(self, image):
        if image.shape[1] == 3:  
            image = self.grayscale_transform(image)

        # Convolve the image with the Sobel kernels
        Gx = F.conv2d(image, self.Kx, padding=1)
        Gy = F.conv2d(image, self.Ky, padding=1)
        
        # Calculate the gradient magnitude
        G = torch.hypot(Gx, Gy)
        G = torch.sigmoid(G)

        # Generate markers using Voronoi propagation
        markers = self.voronoi_propagation(image)
        
        # rnn_input = torch.cat([markers, G], dim=1)
        # segmentation = self.rnn_flooding(rnn_input)
        return markers
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_size = 224
    batch_size = 1
    num_channels = 1
    
    # Create a simple synthetic image with two circles
    img = torch.zeros((batch_size, num_channels, img_size, img_size), dtype=torch.float32)
    img[0, 0, 40:60, 40:60] = 1
    img[0, 0, 70:90, 70:90] = 1
    
    learnable_watershed = LearnableWatershedWithRNN(num_markers=3)
    segmentation = learnable_watershed(img)
    segmentation = segmentation.argmax(dim=1, keepdim=True)
    
    segmentation_np = segmentation.squeeze().detach().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Result")
    plt.imshow(segmentation_np, cmap='jet')
    plt.show()