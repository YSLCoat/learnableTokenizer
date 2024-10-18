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

        # Linear layer to project centroid coordinates to positional embeddings
        self.positional_embedding = nn.Linear(2, embed_dim)
    
        self.feature_proj = nn.Sequential(
            nn.Conv2d(n_channels, self.embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        
        # self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, img):
        # Get the superpixel segments, centroid coordinates, and U-Net features from the tokenizer
        #gradient_map, centroid_coords, segments, unet_features = self.superpixel_tokenizer(img)
        
        gradient_map, centroid_coords, segments= self.superpixel_tokenizer(img)

        batch_size, n_channels, height, width = img.shape

        #features = unet_features  # features: [B, C_out, H, W]
        features = self.cnn(img)  # features: [B, C, Hf, Wf]
        B, C, Hf, Wf = features.shape

        #features = self.feature_proj(features)

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

        # Ensure centroids_normalized is a float tensor
        centroids_normalized = centroid_coords.clone().float()  # Convert to float

        # Normalize centroid coordinates
        centroids_normalized[:, :, 0] /= float(width)   # x-coordinate normalization
        centroids_normalized[:, :, 1] /= float(height)  # y-coordinate normalization

        # Project centroid coordinates to positional embeddings
        pos_embeddings = self.positional_embedding(centroids_normalized.to(img.device))  # [B, n_centroids, embed_dim]

        # Pad pos_embeddings to match max_segments
        pos_embeddings_padded = torch.zeros(B, self.max_segments, self.embed_dim, device=img.device)
        n_centroids = centroids_normalized.shape[1]
        max_segments = self.max_segments
        if n_centroids > max_segments:
            pos_embeddings_padded = pos_embeddings[:, :max_segments, :]
        else:
            pos_embeddings_padded[:, :n_centroids, :] = pos_embeddings

        # Combine embeddings with positional embeddings
        embeddings = embeddings + pos_embeddings_padded
        #embeddings = self.layer_norm(embeddings)
        
        return embeddings  # Shape: [B, max_segments, embed_dim]

        

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, max_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.embed_dim = self.vit.embed_dim

        # Replace the patch embedding with the superpixel tokenizer
        self.vit.patch_embed = differentiableSuperpixelTokenizer(
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