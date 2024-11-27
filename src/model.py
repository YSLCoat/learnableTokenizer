from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from differentiableWatershed.model import VoronoiPropagation

class MaskedLinear(nn.Module):
    '''Masked linear layer.

    Linear layer that only computes projections for elements in provided 
    attention mask to save on compute. 
    '''
    def __init__(self, in_feat, out_feat, bias=True, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.act = activation if activation else nn.Identity()
        if bias:
            self.linear.bias.data.mul_(1e-3)
        
    def forward(self, x, amask):
        # x: [N_total, in_feat], amask: [N_total], where N_total = B * N
        masked_indices = amask.nonzero(as_tuple=True)[0]
        if masked_indices.numel() == 0:
            # If no elements are masked, return zeros
            out = torch.zeros_like(x)
            return out
        masked_input = x[masked_indices]  # [num_masked, in_feat]
        masked_output = self.act(self.linear(masked_input))  # [num_masked, out_feat]
        out = torch.zeros(x.shape[0], self.linear.out_features, dtype=x.dtype, device=x.device)
        out[masked_indices] = masked_output
        return out

class MaskedMLP(nn.Module):
    '''Masked MLP module.

    An MLP module that supports masks for training and inference with variable 
    numbers of tokens in each batch.
    '''
    def __init__(self, embed_dim, hid_dim):
        super().__init__()
        self.L1 = MaskedLinear(embed_dim, hid_dim, activation=nn.GELU())
        self.L2 = MaskedLinear(hid_dim, embed_dim)
    
    def forward(self, x, amask):
        x = self.L1(x, amask)
        x = self.L2(x, amask)
        return x

class differentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, n_segments, n_channels=3, embed_dim=768, hid_dim=512):
        super().__init__()
        self.superpixel_tokenizer = VoronoiPropagation(n_segments)
        self.n_segments = n_segments
        self.embed_dim = embed_dim

        # CNN backbone to extract feature maps
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Positional embedding projection
        self.positional_embed_proj = nn.Linear(2, embed_dim)

        # Masked MLP for feature aggregation
        self.masked_mlp = MaskedMLP(embed_dim, hid_dim)
        
    def compute_centroid_positions(self, segments, device):
        B, H, W = segments.shape
        n_segments = self.n_segments

        # Generate pixel coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        coords = torch.stack((x_coords, y_coords), dim=-1).view(-1, 2)  # [H*W, 2]

        # Flatten segments
        segments_flat = segments.view(B, -1)  # [B, H*W]

        # Expand coords to match batch size
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]

        # Compute centroids using scatter_mean
        centroids = torch_scatter.scatter_mean(
            coords, segments_flat.unsqueeze(-1).expand(-1, -1, 2), dim=1, dim_size=n_segments
        )  # [B, n_segments, 2]

        return centroids

    def forward(self, img):
        # Get the superpixel segments
        _, _, segments = self.superpixel_tokenizer(img)  # segments: [B, H, W]

        # Compute centroids for positional embeddings
        centroids = self.compute_centroid_positions(segments, device=img.device)  # [B, n_segments, 2]

        # Normalize centroids to [0, 1] range
        height, width = img.shape[2], img.shape[3]
        centroids_normalized = centroids / torch.tensor([width, height], device=img.device)  # [B, n_segments, 2]

        # Project centroids to embed_dim
        positional_embeddings = self.positional_embed_proj(centroids_normalized)  # [B, n_segments, embed_dim]

        # Process the image with CNN to get feature maps
        features = self.feature_extractor(img)  # [B, embed_dim, Hf, Wf]
        B, C, Hf, Wf = features.shape
        N = Hf * Wf

        # Downsample segments to match feature map size
        segments = F.interpolate(segments.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()  # [B, Hf, Wf]
        segments = segments.view(B, N)  # [B, N]

        # Reshape features
        features = features.view(B, N, C)  # [B, N, C]

        # Flatten batch and pixels for processing
        features_flat = features.view(B * N, C)  # [B*N, C]
        segments_flat = segments.view(B * N)  # [B*N]
        batch_indices = torch.arange(B, device=img.device).unsqueeze(1).expand(B, N).reshape(B * N)  # [B*N]
        global_superpixel_ids = batch_indices * self.n_segments + segments_flat  # [B*N]
        num_superpixels = B * self.n_segments

        # Apply masked MLP
        mlp_output = self.masked_mlp(features_flat, segments_flat >= 0)  # [B*N, embed_dim]

        # Group features by superpixel ID and retain all pixel embeddings as tokens
        tokens = torch.zeros((num_superpixels, self.embed_dim), device=img.device)
        tokens[global_superpixel_ids] = mlp_output  # Place pixel embeddings in the correct superpixel group

        # Reshape to match ViT input: [B, n_segments, embed_dim]
        tokens = tokens.view(B, self.n_segments, self.embed_dim)

        # Add positional embeddings
        tokens += positional_embeddings

        return tokens  # Shape: [B, n_segments, embed_dim]

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, n_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.vit.num_tokens = n_segments + 1  # Update the number of tokens
        self.embed_dim = self.vit.embed_dim

        # Replace the patch embedding with the superpixel tokenizer
        self.vit.patch_embed = differentiableSuperpixelTokenizer(
            n_segments=n_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim
        )

        # Initialize the class token
        self.vit.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.vit.cls_token, std=.02)

        # Initialize positional embeddings
        self.vit.pos_embed = nn.Parameter(torch.zeros(1, n_segments + 1, self.embed_dim))
        nn.init.trunc_normal_(self.vit.pos_embed, std=.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)  # x: [B, n_segments, embed_dim]

        # Get cls_token and pos_embed
        cls_token = self.vit.cls_token.expand(x.size(0), -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, n_segments + 1, embed_dim]

        # Ensure pos_embed matches [1, n_segments + 1, embed_dim]
        if self.vit.pos_embed.size(1) != x.size(1):
            # Reinitialize pos_embed to match the new sequence length
            self.vit.pos_embed = nn.Parameter(torch.zeros(1, x.size(1), self.embed_dim))
            nn.init.trunc_normal_(self.vit.pos_embed, std=.02)

        x = x + self.vit.pos_embed  # Add positional embeddings

        x = self.vit.pos_drop(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.vit.forward_head(x, pre_logits=False)
        return x
    