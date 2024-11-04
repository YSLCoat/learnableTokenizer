import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation

class differentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, max_segments, n_channels=3, embed_dim=768):
        super().__init__()
        self.superpixel_tokenizer = VoronoiPropagation(max_segments)
        self.max_segments = max_segments
        self.embed_dim = embed_dim

        # Learnable 2D positional embedding grid
        self.pos_embedding_grid = nn.Parameter(
            torch.zeros(1, embed_dim, int(torch.sqrt(torch.tensor(max_segments)).item()), int(torch.sqrt(torch.tensor(max_segments)).item()))
        )
        nn.init.trunc_normal_(self.pos_embedding_grid, std=0.02)
    
        self.feature_proj = nn.Sequential(
            nn.Conv2d(n_channels, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, img):
        # Get the superpixel segments, centroid coordinates, and U-Net features from the tokenizer
        gradient_map, centroid_coords, segments, unet_features = self.superpixel_tokenizer(img)

        batch_size, n_channels, height, width = img.shape

        features = unet_features  # features: [B, C_out, H, W]
        B, C, Hf, Wf = features.shape

        features = self.feature_proj(features)

        # Downsample segments to match feature map size if necessary
        if (Hf, Wf) != segments.shape[1:]:
            segments = F.interpolate(segments.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()

        # Flatten features and segments
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)  # [B * Hf * Wf, embed_dim]
        segments_flat = segments.view(-1)  # [B * Hf * Wf]

        batch_indices = torch.arange(B, device=img.device).unsqueeze(1).expand(B, Hf * Wf).reshape(-1)

        # Compute unique segment IDs per batch
        unique_segment_ids = batch_indices * self.max_segments + segments_flat  # [B * Hf * Wf]

        # Compute per-segment embeddings using scatter_mean
        dim_size = B * self.max_segments
        embeddings = scatter_mean(features_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        embeddings = embeddings.view(B, self.max_segments, self.embed_dim)

        # Prepare centroid coordinates for sampling positional embeddings
        centroids_normalized = centroid_coords.clone().float()
        centroids_normalized[:, :, 0] = 2.0 * (centroids_normalized[:, :, 0] / (height - 1)) - 1.0  # y-coordinate normalization
        centroids_normalized[:, :, 1] = 2.0 * (centroids_normalized[:, :, 1] / (width - 1)) - 1.0   # x-coordinate normalization

        # Create a grid for grid_sample
        centroids_grid = centroids_normalized.unsqueeze(2)  # Shape: [B, max_segments, 1, 2]
        centroids_grid = centroids_grid[..., [1, 0]]  # Swap x and y to match grid_sample ordering

        # Sample positional embeddings from the grid
        pos_embeddings = F.grid_sample(
            self.pos_embedding_grid.expand(B, -1, -1, -1).to(img.device),  # Shape: [B, embed_dim, grid_size, grid_size]
            centroids_grid.to(img.device),  # Corrected shape: [B, max_segments, 1, 2]
            mode='bilinear',
            align_corners=True
        ).squeeze(3).transpose(1, 2)  # Shape after squeeze and transpose: [B, max_segments, embed_dim]


        # Combine embeddings with positional embeddings
        embeddings = embeddings + pos_embeddings
        embeddings = self.layer_norm(embeddings)

        return embeddings  # Shape: [B, max_segments, embed_dim]

        

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, max_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.embed_dim = self.vit.embed_dim
        self.max_segments = max_segments
        
        # self.vit.drop_rate = 0.1  # Adjust as needed
        # self.vit.attn_drop_rate = 0.1  # Adjust as needed
        # self.vit.drop_path_rate = 0.1  # Adjust as needed

        # Replace the patch embedding with the superpixel tokenizer
        self.vit.patch_embed = differentiableSuperpixelTokenizer(
            max_segments=max_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim
        )

        self.vit.num_tokens = max_segments + 1  # Update the number of tokens

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.vit.patch_embed(x)  # [B, max_segments, embed_dim]
        b, n, d = embeddings.shape

        cls_tokens = self.vit.cls_token.expand(b, -1, -1)  # [B, 1, D]

        x = torch.cat((cls_tokens, embeddings), dim=1)  # [B, n+1, D]

        x = self.vit.pos_drop(x)
        x = self.vit.norm_pre(x)

        if self.vit.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.vit.blocks, x)
        else:
            x = self.vit.blocks(x)

        x = self.vit.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.vit.forward_head(x, pre_logits=False)
        return x

    
