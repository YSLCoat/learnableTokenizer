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
        x = x.view(batch_size, h * w, channels) 
        h_t = None  
        for _ in range(self.num_iterations):
            x, h_t = self.rnn(x, h_t)  
            x = F.relu(x)
        
        x = x.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # Reshape back to (batch_size, channels, h, w)
        x = self.output_layer(x)
        return x

class LearnableWatershedWithRNN(nn.Module):
    def __init__(self, num_markers=3, num_iterations=20, rnn_hidden_channels=51):
        super(LearnableWatershedWithRNN, self).__init__()

        self.num_markers = num_markers

        # Convolutional layers for edge detection
        self.edge_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.edge_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.edge_conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # Output is a single channel for edge detection

        # Convolutional layers for marker prediction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_markers, kernel_size=3, padding=1)  # Output channels = num_markers
        
        self.num_classes = num_markers
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)  # No need to send to device, done in forward pass

        # RNN for flooding approximation
        self.rnn_flooding = FloodingRNN(in_channels=51,  # Markers and gradient magnitude
                                        hidden_channels=rnn_hidden_channels, 
                                        out_channels=num_markers, 
                                        num_iterations=num_iterations)
        
    def forward(self, image):
        if image.shape[1] == 3:  
            image = self.grayscale_transform(image)

        # Edge detection using separate convolutional layers
        edge_x = F.relu(self.edge_conv1(image))
        edge_x = F.relu(self.edge_conv2(edge_x))
        G = torch.sigmoid(self.edge_conv3(edge_x))  # Use sigmoid to get values in range [0, 1] for gradient magnitude
        
        G = F.normalize(G, p=2, dim=(-2, -1), eps=1e-7)

        # Generate markers using the learnable marker generator
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        x = F.softmax(x, dim=1)
        
        # def gumbel_softmax(logits, tau=1.0, hard=False):
        #     gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        #     gumbels = (logits + gumbels) / tau  # logits + gumbels for sampling
        #     y_soft = gumbels.softmax(dim=1)

        #     if hard:
        #         # Straight-through trick for discrete sampling
        #         index = y_soft.max(dim=1, keepdim=True)[1]
        #         y_hard = torch.zeros_like(logits).scatter_(1, index, 1.0)
        #         return (y_hard - y_soft).detach() + y_soft
        #     else:
        #         return y_soft

        # # Apply Gumbel-Softmax to the output of the CNN
        # x = gumbel_softmax(x, tau=1.0, hard=True)  # Now x contains differentiable approximations to one-hot encoded markers
        # assert 0, x.shape
        # Pass to the RNN
        
        
        _, markers = torch.max(x, dim=1, keepdim=True)
        # markers = torch.argmax(x, dim=1, keepdim=True)
        # Prepare input for RNN (concatenate markers and gradient)
        rnn_input = torch.cat([x, G], dim=1).to(device)
        
        
        
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