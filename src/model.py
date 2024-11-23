from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from differentiableWatershed.model import VoronoiPropagation

class differentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, n_segments, n_channels=3, embed_dim=768):
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

        # Positional embedding for superpixel masks
        self.positional_embedding_conv = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1)

        # Attention mechanism for feature aggregation
        self.superpixel_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, img):
        # Get the superpixel segments and centroid coordinates from the tokenizer
        _, _, segments = self.superpixel_tokenizer(img)  # segments: [B, H, W]

        batch_size, n_channels, height, width = img.shape

        # Process the image with CNN to get feature maps
        features = self.feature_extractor(img)  # features: [B, embed_dim, Hf, Wf]
        B, C, Hf, Wf = features.shape

        # Downsample segments to match feature map size
        segments = F.interpolate(segments.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()
        # segments: [B, Hf, Wf]

        # Reshape features and segments
        features = features.view(B, C, Hf * Wf).permute(0, 2, 1)  # [B, N, C]
        segments = segments.view(B, Hf * Wf)  # [B, N]

        embeddings = []

        for b in range(B):
            batch_embeddings = []
            for s in range(self.n_segments):
                # Get indices of pixels belonging to the current superpixel
                mask = (segments[b] == s)  # [N]
                pixel_indices = mask.nonzero(as_tuple=True)[0]  # [num_pixels_in_superpixel]

                if pixel_indices.numel() == 0:
                    # If the superpixel is empty, append a zero embedding
                    batch_embeddings.append(torch.zeros(self.embed_dim, device=img.device))
                    continue

                # Extract features for the current superpixel
                pixel_features = features[b, pixel_indices, :]  # [num_pixels_in_superpixel, C]

                # Create superpixel mask for positional embedding
                superpixel_mask = torch.zeros(1, Hf * Wf, device=img.device)
                superpixel_mask[0, pixel_indices] = 1
                superpixel_mask = superpixel_mask.view(1, 1, Hf, Wf)  # [1, 1, Hf, Wf]

                # Compute positional embedding for the superpixel
                pos_embed = self.positional_embedding_conv(superpixel_mask)  # [1, C, Hf, Wf]
                pos_embed = pos_embed.view(1, C, Hf * Wf).permute(0, 2, 1)  # [1, N, C]
                pos_embed = pos_embed[:, pixel_indices, :]  # [1, num_pixels_in_superpixel, C]
                pos_embed = pos_embed.squeeze(0)  # [num_pixels_in_superpixel, C]

                # Add positional embeddings to features
                pixel_features += pos_embed

                # Apply attention (self-attention within the superpixel)
                attn_output, _ = self.superpixel_attention(
                    pixel_features.unsqueeze(0),  # [1, num_pixels_in_superpixel, C]
                    pixel_features.unsqueeze(0),
                    pixel_features.unsqueeze(0)
                )  # [1, num_pixels_in_superpixel, C]

                # Aggregate the attended features (e.g., via mean)
                superpixel_embedding = attn_output.mean(dim=1).squeeze(0)  # [C]
                batch_embeddings.append(superpixel_embedding)

            # Stack embeddings for all superpixels in the batch
            batch_embeddings = torch.stack(batch_embeddings, dim=0)  # [n_segments, C]
            embeddings.append(batch_embeddings)

        embeddings = torch.stack(embeddings, dim=0)  # [B, n_segments, C]

        # Return embeddings without attention mask
        return embeddings  # Shape: [B, n_segments, embed_dim]

        

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, n_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.embed_dim = self.vit.embed_dim

        # Replace the patch embedding with the superpixel tokenizer
        self.vit.patch_embed = differentiableSuperpixelTokenizer(
            n_segments=n_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim
        )

        # Remove positional embeddings from the ViT
        self.vit.pos_embed = None  # Positional embeddings are added in the tokenizer
        self.vit.num_tokens = n_segments + 1  # Update the number of tokens

        # Optionally, add positional embedding for the CLS token
        self.cls_positional_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.vit.patch_embed(x)  # [B, n_segments, embed_dim]
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