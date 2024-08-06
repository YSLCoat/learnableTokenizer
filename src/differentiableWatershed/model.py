import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# SDF func
# def compute_sdf(marker):
#     # Compute the distance transform for foreground (marker)
#     fg_dist = torch.cdist(marker.float(), torch.ones_like(marker.float()))
#     # Compute the distance transform for background (1 - marker)
#     bg_dist = torch.cdist(1 - marker.float(), torch.ones_like(marker.float()))
#     # The SDF is the difference between the background distance and the foreground distance
#     sdf = bg_dist - fg_dist
#     return sdf

class LearnableSobelFilter(nn.Module): # Starting with learnable sobel filter, might go over to fully learnable edge detection function later
    def __init__(self):
        super(LearnableSobelFilter, self).__init__()
        self.Kx = nn.Parameter(torch.tensor([[-1, 0, 1], 
                                             [-2, 0, 2], 
                                             [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        self.Ky = nn.Parameter(torch.tensor([[ 1,  2,  1], 
                                             [ 0,  0,  0], 
                                             [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, image):
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
    
        Gx = F.conv2d(image, self.Kx, padding=1)
        Gy = F.conv2d(image, self.Ky, padding=1)
        
        G = torch.hypot(Gx, Gy)
        #should we normalize the gradient?
        return G

class LearnableMarkerGenerator(nn.Module): # relatively simple boilerplate code for predicting marker location, experiment with U-net or something
    def __init__(self, num_markers=3):
        super(LearnableMarkerGenerator, self).__init__()
        self.num_markers = num_markers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_markers, kernel_size=3, padding=1)  # Output channels = num_markers

    def forward(self, image):
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        x = F.softmax(x, dim=1)
        
        markers = torch.argmax(x, dim=1, keepdim=True)
        
        return markers

class DifferentiableFloodingProcessWithSDF(nn.Module):
    """
    Differentiable Flooding Process with Signed Distance Function (SDF)

    This module performs a differentiable flooding process, similar to the watershed algorithm, 
    to propagate marker information across an image. The process uses a combination of 
    gradient information, marker affinities, and signed distance functions (SDF) to iteratively 
    update label probabilities.

    Args:
        num_classes (int): The number of distinct marker classes.
        num_iterations (int, optional): The number of iterations to perform the flooding process. Default is 10.

    Methods:
        forward(gradient, markers):
            Forward pass to perform the differentiable flooding process.
            
            Args:
                gradient (torch.Tensor): A tensor of shape (batch_size, 1, height, width) representing the gradient magnitudes of the input image.
                markers (torch.Tensor): A tensor of shape (batch_size, num_markers, height, width) representing the initial marker locations.
                
            Returns:
                torch.Tensor: A tensor of shape (batch_size, 1, height, width) containing the final labeled image after the flooding process.
    """
    def __init__(self, num_classes, num_iterations=10):
        super(DifferentiableFloodingProcessWithSDF, self).__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
    
    def forward(self, gradient, markers):
        # b, c, h, w = gradient.size()
        markers_one_hot = F.one_hot(markers.squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        """
        The markers are converted to one-hot encoding. 
        The squeeze(1) removes the channel dimension from markers, one_hot converts the integer labels to one-hot encoded format, 
        and permute(0, 3, 1, 2) rearranges the dimensions to match the input format. 
        The result is a tensor of shape (b, num_classes, h, w).
        """
        
        
        labeled = markers_one_hot.clone()
        sdf = torch.stack([compute_sdf(markers[:, i, :, :]) for i in range(markers.size(1))], dim=1)
        
        for _ in range(self.num_iterations):
            affinity = F.avg_pool2d(labeled, kernel_size=3, stride=1, padding=1)
            labeled = F.softmax(affinity + gradient + sdf, dim=1)
            
        """
        The flooding process iteratively updates the labeled tensor:

        Affinity Calculation: avg_pool2d computes the average value in a 3x3 neighborhood for each pixel. This step promotes the spread of label information by smoothing.
        Label Update: The labeled tensor is updated by combining the affinity map, the gradient, and the SDF, followed by applying the softmax function. 
        The softmax ensures the outputs are probabilities, facilitating the propagation of labels influenced by gradient and SDF information.
        """
        
        labeled = torch.argmax(labeled, dim=1, keepdim=True)
        
        return labeled


# Integrating this into the LearnableWatershed class:

class LearnableWatershedWithSDF(nn.Module):
    def __init__(self, num_markers=3, num_iterations=50):
        super(LearnableWatershedWithSDF, self).__init__()
        self.sobel_filter = LearnableSobelFilter()
        self.marker_generator = LearnableMarkerGenerator(num_markers=num_markers)
        self.flooding_process = DifferentiableFloodingProcessWithSDF(num_classes=num_markers, num_iterations=num_iterations)
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)

    def forward(self, image):
        if image.shape[1] == 3:  # assuming image shape is (batch_size, channels, height, width)
            image = self.grayscale_transform(image)
        # Compute gradient magnitude using the learnable Sobel filter
        gradient = self.sobel_filter(image)

        # Generate markers using the learnable marker generator
        markers = self.marker_generator(image)

        # Perform the differentiable flooding process with SDF
        segmentation = self.flooding_process(gradient, markers)
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
    
    learnable_watershed = LearnableWatershedWithSDF(num_markers=3)
    segmentation = learnable_watershed(img)
    
    print(segmentation.shape)
    
    segmentation_np = segmentation.squeeze().detach().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Result")
    plt.imshow(segmentation_np, cmap='jet')
    plt.show()