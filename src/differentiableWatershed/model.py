import torchvision.transforms as transforms
import torch.nn as nn
import torch 
import torch.functional as F

class LearnableSobelFilter(nn.Module):
    def __init__(self):
        super(LearnableSobelFilter, self).__init__()
        # Initiate learnable parameters with sobel coefficients
        self.Kx = nn.Parameter(torch.tensor([[-1, 0, 1], 
                                             [-2, 0, 2], 
                                             [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        self.Ky = nn.Parameter(torch.tensor([[ 1,  2,  1], 
                                             [ 0,  0,  0], 
                                             [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, image):
        # Dynamic shapes for testing
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        Gx = F.conv2d(image, self.Kx, padding=1)
        Gy = F.conv2d(image, self.Ky, padding=1)
        
        # Calculate the gradient magnitude
        G = torch.hypot(Gx, Gy)
        
        return G

class LearnableMarkerGenerator(nn.Module):
    def __init__(self, num_markers=3):
        super(LearnableMarkerGenerator, self).__init__()
        self.num_markers = num_markers
        # Architecture subject to change
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_markers, kernel_size=3, padding=1)  # Output channels = num_markers

    def forward(self, image):
        # Dynamic shapes for testing
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        # softmax to get probabilities for each marker
        x = F.softmax(x, dim=1)
        
        # Convert probabilities to marker indices
        markers = torch.argmax(x, dim=1, keepdim=True)
        
        return markers
    
class DifferentiableFloodingProcess(nn.Module):
    def __init__(self, num_classes, num_iterations=10):
        super(DifferentiableFloodingProcess, self).__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
    
    def forward(self, gradient, markers):
        b, c, h, w = gradient.size()
        
        # one-hot encoded markers
        markers_one_hot = F.one_hot(markers.squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Initialize the labeled image with markers
        labeled = markers_one_hot.clone()
        
        for _ in range(self.num_iterations):
            # Compute the affinity between neighboring pixels
            affinity = F.avg_pool2d(labeled, kernel_size=3, stride=1, padding=1)
            
            # Update the labels using softmax for differentiability
            labeled = F.softmax(affinity + gradient, dim=1)
        
        # Convert the labeled tensor to final segmentation by using argmax
        labeled = torch.argmax(labeled, dim=1, keepdim=True)
        
        return labeled


class differentiableWatershed(nn.Module):
    def __init__(self, num_markers=3):
        super(differentiableWatershed, self).__init__()
        self.sobel_filter = LearnableSobelFilter()
        self.marker_generator = LearnableMarkerGenerator(num_markers=num_markers)
        self.flooding_process = DifferentiableFloodingProcess(num_classes=num_markers)
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)

    def forward(self, image):
        if image.shape[1] == 3:  # assuming image shape is (batch_size, channels, height, width)
            image = self.grayscale_transform(image)
        gradient = self.sobel_filter(image)
        markers = self.marker_generator(image)
        segmentation = self.flooding_process(gradient, markers)
        return segmentation
