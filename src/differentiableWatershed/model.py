import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch_geometric.nn import GCNConv

import torch
from torch_geometric.data import Data
device = "cuda" if torch.cuda.is_available() else "cpu"

class FloodingRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_iterations):
        super(FloodingRNN, self).__init__()
        self.num_iterations = num_iterations
        self.rnn = nn.GRU(in_channels, hidden_channels, batch_first=True)
        self.output_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.view(batch_size, h * w, channels)  # Flatten the spatial dimensions
        h_t = None  # Hidden state initialized to None
        for _ in range(self.num_iterations):
            x, h_t = self.rnn(x, h_t)  # Process the sequence
            x = F.relu(x)
        
        x = x.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # Reshape back to (batch_size, channels, h, w)
        x = self.output_layer(x)
        return x

class LearnableWatershedWithRNN(nn.Module):
    def __init__(self, num_markers=3, num_iterations=50, rnn_hidden_channels=2):
        super(LearnableWatershedWithRNN, self).__init__()
        self.Kx = nn.Parameter(torch.tensor([[-1, 0, 1], 
                                             [-2, 0, 2], 
                                             [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        self.Ky = nn.Parameter(torch.tensor([[ 1,  2,  1], 
                                             [ 0,  0,  0], 
                                             [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        self.num_markers = num_markers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_markers, kernel_size=3, padding=1)  # Output channels = num_markers
        
        self.num_classes = num_markers
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1).to(device)

        # RNN for flooding approximation
        self.rnn_flooding = FloodingRNN(in_channels=2,  # Markers and gradient magnitude
                                        hidden_channels=rnn_hidden_channels, 
                                        out_channels=num_markers, 
                                        num_iterations=num_iterations).to(device)
        
    def forward(self, image):
        if image.shape[1] == 3:  # assuming image shape is (batch_size, channels, height, width)
            image = self.grayscale_transform(image)
        
        # Compute gradient magnitude using the learnable Sobel filter
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
    
        Gx = F.conv2d(image, self.Kx, padding=1)
        Gy = F.conv2d(image, self.Ky, padding=1)
        
        G = torch.hypot(Gx, Gy)

        # Generate markers using the learnable marker generator
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        x = F.softmax(x, dim=1)
        
        _, markers = torch.max(x, dim=1, keepdim=True)
        # markers = torch.argmax(x, dim=1, keepdim=True)
        
        # Prepare input for RNN (concatenate markers and gradient)
        rnn_input = torch.cat([markers.float(), G], dim=1).to(device)
        # Run the RNN to approximate the flooding process
        segmentation = self.rnn_flooding(rnn_input)   
        
        return segmentation
    
    
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