import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation, BoundaryPathFinder

class differentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, max_segments, n_channels=3, embed_dim=768, superpixel_algorithm='voronoi'):
        super().__init__()
        if superpixel_algorithm == 'voronoi':
            self.superpixel_tokenizer = VoronoiPropagation(max_segments, n_channels)
        elif superpixel_algorithm == 'pathfinder':
            #num_segments = torch.sqrt()
            self.superpixel_tokenizer = BoundaryPathFinder(16, 16)
        self.max_segments = max_segments
        self.embed_dim = embed_dim
    
        self.feature_proj = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, self.embed_dim)
        )
        
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

        # Compute positional embeddings from centroids
        centroids_normalized = centroid_coords.clone().float()
        centroids_normalized[:, :, 0] /= (height - 1)  # y-coordinate
        centroids_normalized[:, :, 1] /= (width - 1)   # x-coordinate
        pos_embeddings = self.pos_mlp(centroids_normalized.to(img.device))  # [B, max_segments, embed_dim]

        return embeddings, pos_embeddings  # Shape: [B, max_segments, embed_dim]

        

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

        # Update positional embeddings to match the new number of tokens
        # self.vit.pos_embed = nn.Parameter(
        #     torch.zeros(1, max_segments + 1, self.embed_dim)
        # )
        # nn.init.trunc_normal_(self.vit.pos_embed, std=0.02)
        
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        self.vit.num_tokens = max_segments + 1  # Update the number of tokens

        # Remove the CLS positional embedding if it exists
        if hasattr(self.vit, 'cls_positional_embedding'):
            del self.vit.cls_positional_embedding

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings, pos_embeddings = self.vit.patch_embed(x)  # [B, max_segments, embed_dim]
        b, n, d = embeddings.shape

        cls_tokens = self.vit.cls_token.expand(b, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, embeddings), dim=1)  # [B, n+1, D]
        
        cls_pos_embed = self.cls_pos_embed.expand(b, -1, -1)  # [B, 1, D]
        pos_embed = torch.cat((cls_pos_embed, pos_embeddings), dim=1)  # [B, n+1, D]

        # Add positional embeddings
        x = x + pos_embed
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

    
