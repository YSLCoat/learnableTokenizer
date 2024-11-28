import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import timm
from differentiableWatershed.model import VoronoiPropagation

class DifferentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, n_channels, n_segments, embed_dim=768, box_size=16):
        super().__init__()
        self.n_channels = n_channels
        self.n_segments = n_segments
        self.embed_dim = embed_dim
        self.box_size = box_size
        self.superpixel_tokenizer = VoronoiPropagation(n_segments)
        # Convolutional layer without activation function
        self.conv = nn.Conv2d(n_channels, embed_dim, kernel_size=box_size, bias=False)

    def forward(self, img):
        B, C, H, W = img.size()
        device = img.device

        # Generate superpixel segments
        _, _, segments = self.superpixel_tokenizer(img)  # segments: [B, H, W]

        # Create one-hot masks for superpixels
        masks = torch.nn.functional.one_hot(segments, num_classes=self.n_segments).permute(0, 3, 1, 2).float()  # [B, n_segments, H, W]

        # Compute coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        y_coords = y_coords.float()
        x_coords = x_coords.float()

        # Compute centroid coordinates
        sum_x = (masks * x_coords).view(B, self.n_segments, -1).sum(dim=2)  # [B, n_segments]
        sum_y = (masks * y_coords).view(B, self.n_segments, -1).sum(dim=2)
        pixel_counts = masks.view(B, self.n_segments, -1).sum(dim=2)  # [B, n_segments]
        pixel_counts = torch.clamp(pixel_counts, min=1e-6)  # Avoid division by zero
        x_c = sum_x / pixel_counts  # [B, n_segments]
        y_c = sum_y / pixel_counts  # [B, n_segments]

        # Pad the images
        box_half_size = self.box_size // 2
        pad = (box_half_size, box_half_size, box_half_size, box_half_size)  # Left, Right, Top, Bottom
        img_padded = F.pad(img, pad, mode='constant', value=0)

        # Adjust centroid coordinates due to padding
        x_c_padded = x_c + box_half_size
        y_c_padded = y_c + box_half_size

        # Compute bounding box coordinates
        x_min = (x_c_padded - box_half_size).long()  # [B, n_segments]
        y_min = (y_c_padded - box_half_size).long()

        # Flatten indices
        batch_indices = torch.arange(B, device=device).unsqueeze(1).repeat(1, self.n_segments).view(-1)  # [B * n_segments]
        x_min_flat = x_min.view(-1)
        y_min_flat = y_min.view(-1)

        # Initialize tensor for patches
        N_patches = B * self.n_segments
        patches = torch.zeros(N_patches, C, self.box_size, self.box_size, device=device)

        # Extract patches
        for idx in range(N_patches):
            b = batch_indices[idx]
            x0 = x_min_flat[idx]
            y0 = y_min_flat[idx]
            patch = img_padded[b, :, y0:y0+self.box_size, x0:x0+self.box_size]  # [C, box_size, box_size]
            patches[idx] = patch

        # Apply convolutional layer
        embeddings = self.conv(patches)  # [N_patches, embed_dim, 1, 1]
        embeddings = embeddings.view(B, self.n_segments, self.embed_dim)

        return embeddings  # [B, n_segments, embed_dim]
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
