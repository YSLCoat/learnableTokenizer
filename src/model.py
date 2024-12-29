import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max

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

        # Linear layer to project mean an max embeddings of superpixels to same embedding space       
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, img):
        # Tokenize into superpixels
        gradient_map, centroid_coords, segments = self.superpixel_tokenizer(img)
        # segments: [B, H, W]
        # centroid_coords: [B, n_centroids, 2]

        batch_size, n_channels, height, width = img.shape

        # Extract features
        features = self.cnn(img)  # [B, C, Hf, Wf]
        B, C, Hf, Wf = features.shape

        # Flatten features and segments
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*Hf*Wf, C]
        segments_flat = segments.view(-1)  # [B*Hf*Wf]

        # Create batch indices to offset segment IDs per batch
        batch_indices = torch.arange(B, device=img.device).unsqueeze(1)  # [B, 1]
        batch_indices = batch_indices.expand(B, Hf * Wf).reshape(-1)     # [B*Hf*Wf]

        # Shift segment IDs so each batch has unique ID ranges:
        # segment_id ∈ [0..max_segments - 1], offset by batch_idx * max_segments
        unique_segment_ids = batch_indices * self.max_segments + segments_flat  # [B * Hf * Wf]

        # We'll have B * max_segments possible slots
        dim_size = B * self.max_segments

        # -- 1) Compute scatter_mean --
        embeddings_mean = scatter_mean(features_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        embeddings_mean = embeddings_mean.view(B, self.max_segments, C)  # [B, max_segments, C]

        # -- 2) Compute scatter_max --
        # scatter_max returns a tuple (values, indices). We only need the values here.
        embeddings_max, _ = scatter_max(features_flat, unique_segment_ids, dim=0, dim_size=dim_size)
        embeddings_max = embeddings_max.view(B, self.max_segments, C)  # [B, max_segments, C]

        # -- 3) Concatenate mean & max embeddings --
        embeddings_concat = torch.cat([embeddings_mean, embeddings_max], dim=-1)  
        # [B, max_segments, 2*C]

        # -- 4) Fuse the concatenated embeddings back to size embed_dim --
        embeddings_fused = self.fusion(embeddings_concat)  # [B, max_segments, embed_dim]

        # -- 5) Positional embeddings from superpixel centroids --
        centroids_normalized = centroid_coords.clone().float()  
        centroids_normalized[:, :, 0] /= float(width)   # x / width
        centroids_normalized[:, :, 1] /= float(height)  # y / height

        pos_embeddings = self.positional_embedding(centroids_normalized.to(img.device)) 
        # [B, n_centroids, embed_dim]

        # If the number of centroids < max_segments, we need to pad
        n_centroids = pos_embeddings.shape[1]
        pos_embeddings_padded = torch.zeros(B, self.max_segments, self.embed_dim, device=img.device)

        if n_centroids > self.max_segments:
            # Truncate if there are more centroids than max_segments
            pos_embeddings_padded = pos_embeddings[:, :self.max_segments, :]
        else:
            # Otherwise, copy and leave the rest zero
            pos_embeddings_padded[:, :n_centroids, :] = pos_embeddings

        # -- 6) Add positional embeddings to the fused embeddings --
        final_embeddings = embeddings_fused + pos_embeddings_padded  # [B, max_segments, embed_dim]

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