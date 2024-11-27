import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import timm
from differentiableWatershed.model import VoronoiPropagation

class DifferentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, n_channels, n_segments, embed_dim=768):
        super().__init__()
        self.n_segments = n_segments
        self.embed_dim = embed_dim
        self.superpixel_tokenizer = VoronoiPropagation(n_segments)
        # Update the linear layer to account for positional information (C + 2)
        self.linear = nn.Linear(n_channels + 2, embed_dim)

    def forward(self, img):
        B, C, H, W = img.size()
        device = img.device
        N_pixels = H * W

        # Generate superpixel segments
        _, _, segments = self.superpixel_tokenizer(img)  # segments: [B, H, W]
        segments_flat = segments.view(B, N_pixels)  # [B, N_pixels]

        # Flatten the image tensor
        img_flat = img.view(B, C, N_pixels).permute(0, 2, 1)  # [B, N_pixels, C]

        # Compute normalized positional coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device).float() / (H - 1),
            torch.arange(W, device=device).float() / (W - 1),
            indexing="ij"
        )
        coords = torch.stack((x_coords, y_coords), dim=-1).view(1, N_pixels, 2).expand(B, -1, -1)  # [B, N_pixels, 2]

        # Concatenate pixel features and positional coordinates
        img_flat = torch.cat([img_flat, coords], dim=-1)  # [B, N_pixels, C + 2]

        # Create batch indices
        batch_indices = torch.arange(B, device=device).unsqueeze(1).repeat(1, N_pixels)  # [B, N_pixels]

        # Compute unique segment IDs
        segment_ids = segments_flat + batch_indices * self.n_segments  # [B, N_pixels]

        # Flatten for scatter operations
        img_flat = img_flat.view(B * N_pixels, C + 2)  # [B*N_pixels, C + 2]
        segment_ids = segment_ids.view(B * N_pixels)  # [B*N_pixels]

        # Apply linear projection to pixel features
        pixel_embeddings = self.linear(img_flat)  # [B*N_pixels, embed_dim]

        # Aggregate pixel embeddings per superpixel using scatter_mean
        superpixel_embeddings = torch_scatter.scatter_mean(
            pixel_embeddings, segment_ids, dim=0, dim_size=B * self.n_segments
        )  # [B*n_segments, embed_dim]

        # Reshape to [B, n_segments, embed_dim]
        superpixel_embeddings = superpixel_embeddings.view(B, self.n_segments, self.embed_dim)

        return superpixel_embeddings  # [B, n_segments, embed_dim]

class DifferentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, n_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        self.n_segments = n_segments
        self.embed_dim = None  # Will be set after creating the model

        # Create the ViT model
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # Store the embed_dim from the ViT model
        self.embed_dim = self.vit.embed_dim

        # Replace the patch embedding with the superpixel tokenizer
        self.vit.patch_embed = DifferentiableSuperpixelTokenizer(
            n_channels=num_channels,
            n_segments=n_segments,
            embed_dim=self.embed_dim
        )

        # Adjust the number of tokens (n_segments + 1 for CLS token)
        self.vit.num_tokens = n_segments + 1

        # Initialize the class token
        self.vit.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.vit.cls_token, std=.02)

        # Initialize positional embeddings
        self.vit.pos_embed = nn.Parameter(torch.zeros(1, n_segments + 1, self.embed_dim))
        nn.init.trunc_normal_(self.vit.pos_embed, std=.02)

        # If using a pretrained model, interpolate positional embeddings
        if pretrained:
            self.interpolate_positional_embeddings()

    def interpolate_positional_embeddings(self):
        # Interpolate the pretrained positional embeddings to match n_segments + 1
        pretrained_pos_embed = self.vit.pos_embed  # [1, old_num_tokens, embed_dim]
        old_num_tokens = pretrained_pos_embed.size(1)
        new_num_tokens = self.n_segments + 1  # CLS token + n_segments

        if old_num_tokens != new_num_tokens:
            # Exclude the CLS token from interpolation
            cls_pos_embed = pretrained_pos_embed[:, 0:1, :]  # [1, 1, embed_dim]
            other_pos_embed = pretrained_pos_embed[:, 1:, :]  # [1, old_num_tokens - 1, embed_dim]

            # Interpolate positional embeddings
            new_pos_embed = F.interpolate(
                other_pos_embed.transpose(1, 2),
                size=new_num_tokens - 1,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [1, new_num_tokens - 1, embed_dim]

            # Concatenate CLS token back
            self.vit.pos_embed = nn.Parameter(torch.cat([cls_pos_embed, new_pos_embed], dim=1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)  # x: [B, n_segments, embed_dim]

        # Get cls_token and pos_embed
        cls_token = self.vit.cls_token.expand(x.size(0), -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, n_segments + 1, embed_dim]

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
