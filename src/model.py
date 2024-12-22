import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_min, scatter_add

class DifferentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, max_segments, n_channels=3, embed_dim=768):
        super().__init__()
        self.superpixel_tokenizer = VoronoiPropagation(max_segments)
        self.max_segments = max_segments
        self.embed_dim = embed_dim

        # CNN backbone to extract feature maps
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        # Linear layer to project centroid coordinates to positional embeddings
        self.positional_embedding = nn.Linear(2, embed_dim)

        # Fusion layer to project [mean_emb | max_emb] -> embed_dim
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)

        # ----------------------------------------------------------------------
        # NEW: Embed shape descriptors (e.g. area, aspect_ratio).
        # Feel free to add more features (perimeter, bounding_box_area, etc.)
        # shape_feature_dim = 2 -> [area, aspect_ratio]
        # Then we map shape features -> embed_dim
        # ----------------------------------------------------------------------
        self.shape_feature_dim = 2  
        self.shape_embedding = nn.Sequential(
            nn.Linear(self.shape_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, img):
        """
        Args:
            img: [B, 3, H, W]
        Returns:
            final_embeddings: [B, max_segments, embed_dim]
        """
        # Tokenize into superpixels
        gradient_map, centroid_coords, segments = self.superpixel_tokenizer(img)
        # segments: [B, H, W]
        # centroid_coords: [B, n_centroids, 2]

        B, _, H, W = img.shape

        # -------------------------------------------------
        # 1) Extract CNN features
        # -------------------------------------------------
        features = self.cnn(img)  # [B, C, H, W] if your CNN keeps the same spatial size
        _, C, Hf, Wf = features.shape
        # (We'll assume Hf==H and Wf==W for simplicity.)

        # -------------------------------------------------
        # 2) Flatten features & segments for scatter
        # -------------------------------------------------
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        segments_flat = segments.view(-1)                            # [B*H*W]

        # Create per-batch offsets so superpixel IDs are unique across the batch
        batch_indices = torch.arange(B, device=img.device).unsqueeze(1)  # [B, 1]
        batch_indices = batch_indices.expand(B, Hf * Wf).reshape(-1)     # [B*H*W]
        unique_segment_ids = batch_indices * self.max_segments + segments_flat  # [B*H*W]
        dim_size = B * self.max_segments  # total "slots"

        # -------------------------------------------------
        # 3) Compute scatter_mean and scatter_max
        # -------------------------------------------------
        embeddings_mean = scatter_mean(features_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        embeddings_mean = embeddings_mean.view(B, self.max_segments, C)

        embeddings_max, _ = scatter_max(features_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        embeddings_max = embeddings_max.view(B, self.max_segments, C)

        embeddings_concat = torch.cat([embeddings_mean, embeddings_max], dim=-1)
        embeddings_fused = self.fusion(embeddings_concat)  # [B, max_segments, embed_dim]

        # -------------------------------------------------
        # 4) Compute shape descriptors (area, aspect ratio)
        # -------------------------------------------------
        # 4a) Build pixel coordinate maps [H, W] -> flatten -> broadcast for B if needed
        y_grid = torch.arange(Hf, device=img.device).unsqueeze(1).expand(Hf, Wf)  # [H, W]
        x_grid = torch.arange(Wf, device=img.device).unsqueeze(0).expand(Hf, Wf)  # [H, W]
        y_coords_flat = y_grid.reshape(-1)  # [H*W]
        x_coords_flat = x_grid.reshape(-1)  # [H*W]

        # 4b) For each pixel, we add 1 to the area. So area is scatter_add(1, segment_id).
        area = scatter_add(
            torch.ones_like(segments_flat, dtype=torch.float), 
            unique_segment_ids,
            dim=0,
            dim_size=dim_size
        ).view(B, self.max_segments)

        # 4c) bounding boxes: (min_x, max_x, min_y, max_y)
        min_x, _ = scatter_min(x_coords_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        max_x, _ = scatter_max(x_coords_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        min_y, _ = scatter_min(y_coords_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        max_y, _ = scatter_max(y_coords_flat, unique_segment_ids, dim=0, dim_size=dim_size)

        min_x = min_x.view(B, self.max_segments)
        max_x = max_x.view(B, self.max_segments)
        min_y = min_y.view(B, self.max_segments)
        max_y = max_y.view(B, self.max_segments)

        # 4d) Aspect ratio = (width / height) of the bounding box
        # (Add 1 to avoid zero-size bounding boxes.)
        # Note: You could also do (max_y-min_y+1)/(max_x-min_x+1) if you prefer.
        box_width = (max_x - min_x + 1).clamp(min=1e-6)
        box_height = (max_y - min_y + 1).clamp(min=1e-6)
        aspect_ratio = box_width / box_height  # shape [B, max_segments]

        # Stack shape features -> [B, max_segments, shape_feature_dim]
        shape_feats = torch.stack([area, aspect_ratio], dim=-1)

        # 4e) If the actual number of superpixels is n_centroids < max_segments,
        #     not all slots are used. For uniform handling, we might want
        #     to zero out the shape features for IDs >= n_centroids.
        #     You can do this with a small loop or by using the info in `centroid_coords`.
        #
        #     However, if VoronoiPropagation can produce *up to* max_segments,
        #     typically each ID [0..n_centroids-1] is valid. 
        #     We'll not strictly mask them here, but you could if needed.

        # -------------------------------------------------
        # 5) Embed shape features
        # -------------------------------------------------
        shape_emb = self.shape_embedding(shape_feats)  # [B, max_segments, embed_dim]

        # -------------------------------------------------
        # 6) Positional embeddings from superpixel centroids
        # -------------------------------------------------
        centroids_normalized = centroid_coords.clone().float()
        centroids_normalized[:, :, 0] /= float(W)   # x / width
        centroids_normalized[:, :, 1] /= float(H)   # y / height

        pos_embeddings = self.positional_embedding(centroids_normalized.to(img.device)) 
        # [B, n_centroids, embed_dim]

        n_centroids = pos_embeddings.shape[1]
        pos_embeddings_padded = torch.zeros(B, self.max_segments, self.embed_dim, device=img.device)
        if n_centroids > self.max_segments:
            pos_embeddings_padded = pos_embeddings[:, :self.max_segments, :]
        else:
            pos_embeddings_padded[:, :n_centroids, :] = pos_embeddings

        # -------------------------------------------------
        # 7) Combine everything
        # -------------------------------------------------
        # final_embeddings currently = embeddings_fused + positional
        # We'll add shape_emb as well.
        final_embeddings = embeddings_fused + pos_embeddings_padded + shape_emb
        # [B, max_segments, embed_dim]

        return final_embeddings


        

class DifferentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, max_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)
        self.embed_dim = self.vit.embed_dim

        # Replace the patch embedding with the superpixel tokenizer
        self.vit.patch_embed = DifferentiableSuperpixelTokenizer(
            max_segments=max_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim
        )

        # Remove positional embeddings from the ViT
        self.vit.pos_embed = None  # Positional embeddings are added in the tokenizer
        self.vit.num_tokens = max_segments + 1  # Update the number of tokens

        # Optionally, add positional embedding for the CLS token
        self.cls_positional_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.vit.patch_embed(x)  # [B, max_segments, embed_dim]
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
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.vit.forward_head(x, pre_logits=False)
        return x