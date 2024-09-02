import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
import segmentation_models_pytorch as smp

device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)




class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvGRUCell, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv_gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, kernel_size=kernel_size, padding=padding)
        self.conv_can = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, input_tensor, h_prev):
        combined = torch.cat([input_tensor, h_prev], dim=1)

        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_channels, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        combined_reset = torch.cat([input_tensor, h_prev * reset_gate], dim=1)
        h_new = torch.tanh(self.conv_can(combined_reset))

        h_final = h_prev * (1 - update_gate) + h_new * update_gate

        return h_final

class StackedConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size=3, padding=1):
        super(StackedConvGRU, self).__init__()
        self.num_layers = num_layers

        self.rnn_layers = nn.ModuleList([
            ConvGRUCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size, padding)
            for i in range(num_layers)
        ])

    def forward(self, x):
        batch_size, _, height, width = x.size()
        h = [torch.zeros(batch_size, layer.hidden_channels, height, width, device=x.device) for layer in self.rnn_layers]

        for i, layer in enumerate(self.rnn_layers):
            h[i] = layer(x if i == 0 else h[i-1], h[i])

        return h[-1]  # Return the output of the last layer
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
            sy = sy.float() / (self.num_clusters)
            sx = sx.float() / (self.num_clusters)
        
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
        
        # Compute L2 distance using the centroids
        l2 = gauss2d(y - sy.unsqueeze(1), x - sx.unsqueeze(1), self.std)
        # Softmax to get soft Voronoi regions
        markers = F.softmax(l2, dim=1).view(batch_size, self.num_clusters, self.height, self.width)
        
        return markers

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
        
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)  # No need to send to device, done in forward pass

        # Voronoi propagation module
        self.voronoi_propagation = VoronoiPropagation(num_clusters=num_markers, height=224, width=224)
        
        # ConvRNN module to approximate JFA
        self.conv_rnn = StackedConvGRU(input_channels=num_markers + 1, hidden_channels=rnn_hidden_channels, num_layers=5)
        self.unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=num_markers+4, classes=num_markers).to(device)
        # Final output layer
        self.final_conv = nn.Conv2d(rnn_hidden_channels, num_markers, kernel_size=1)

        
    def forward(self, image):
        if image.shape[1] == 3:  
            image_greyscale_converted = self.grayscale_transform(image)

        # Convolve the image with the Sobel kernels
        Gx = F.conv2d(image_greyscale_converted, self.Kx, padding=1)
        Gy = F.conv2d(image_greyscale_converted, self.Ky, padding=1)
        
        # Calculate the gradient magnitude
        G = torch.hypot(Gx, Gy)
        G = torch.sigmoid(G)
        # Generate markers using Voronoi propagation
        markers = self.voronoi_propagation(image_greyscale_converted)
        concatenated_input = torch.cat((image, markers, G), dim=1)
        # Approximate JFA using ConvRNN
        #refined_markers = self.conv_rnn(concatenated_input)
        #concatenated_input = torch.cat((markers, G), dim=1)
        # Final output after JFA approximation
        #output = self.final_conv(refined_markers)
        
        output = self.unet(concatenated_input)
        
        return output
    
    
    
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