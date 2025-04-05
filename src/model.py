import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from timm.models._manipulate import checkpoint_seq
from differentiableTokenizer.model import VoronoiPropagation, BoundaryPathFinder, SLICSegmentation
from torch_scatter import scatter_mean, scatter_max

class DifferentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, max_segments, n_channels=3, sobel_init=False, embed_dim=768,
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