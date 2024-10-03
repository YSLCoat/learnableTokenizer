import timm
import torch
import torch.nn as nn

from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation

class differentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, max_segments, n_channels=3, embed_dim=768):
        super().__init__()
        self.superpixel_tokenizer = VoronoiPropagation(max_segments)
        self.max_segments = max_segments
        self.embed_dim = embed_dim

        # Linear layer to project raw pixel values to embedding space
        self.feature_projection = nn.Linear(n_channels, embed_dim)

        # Attention layer to pool variable-length pixel embeddings into a fixed-size embedding
        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=4)

    def forward(self, img):
        # Get the superpixel segments from the tokenizer
        _, _, segments = self.superpixel_tokenizer(img)  # segments shape: [batch_size, height, width]
        batch_size, n_channels, height, width = img.shape
        
        # Flatten the image and segment labels for easier indexing
        img_flat = img.view(batch_size, n_channels, -1)  # [batch_size, n_channels, height * width]
        segments_flat = segments.view(batch_size, -1)    # [batch_size, height * width]

        superpixel_features = []

        for segment_id in range(self.max_segments):
            # Create a mask for the current segment
            segment_mask = (segments_flat == segment_id)  # [batch_size, height * width]

            # Check if the segment contains any pixels
            if segment_mask.sum() == 0:
                # Append zeros for this segment if no pixels are found
                superpixel_features.append(torch.zeros(batch_size, self.embed_dim, device=img.device))
                continue

            # Gather the indices of the pixels belonging to the current segment
            pixel_indices = torch.nonzero(segment_mask, as_tuple=True)  # Indices of pixels in the segment

            # Use the indices to gather the pixels in the segment from img_flat
            pixels_in_segment = img_flat[:, :, pixel_indices[1]]  # [batch_size, n_channels, num_pixels_in_segment]

            # Project raw pixel values directly to the embedding space
            projected_pixels = self.feature_projection(pixels_in_segment.permute(2, 0, 1))  # [num_pixels, batch_size, embed_dim]

            # Apply attention to pool variable-length pixel sequences into a fixed-size embedding
            attn_output, _ = self.attention_pool(projected_pixels, projected_pixels, projected_pixels)

            # Pool along the sequence dimension (num_pixels)
            pooled_features = attn_output.mean(dim=0)  # [batch_size, embed_dim]

            # Store the pooled features for this segment
            superpixel_features.append(pooled_features)

        # Stack all the superpixel features into a tensor
        superpixel_features = torch.stack(superpixel_features, dim=1)  # [batch_size, max_segments, embed_dim]

        return superpixel_features

        

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, max_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        
        
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        self.embed_dim = self.vit.embed_dim
        
        self.patch_embed = differentiableSuperpixelTokenizer(
            max_segments=max_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim
        )
        self.vit.patch_embed = self.patch_embed
        self.vit.pos_embed = nn.Parameter(torch.randn(1, max_segments + 1, self.embed_dim) * .02)
        self.vit.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)) 
        
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        b, n, d = x.shape
        cls_tokens = self.vit.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vit.pos_embed[:, :] # n is the number of segments
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        if self.vit.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.vit.attn_pool is not None:
            x = self.vit.attn_pool(x)
        elif self.vit.global_pool == 'avg':
            x = x[:, self.vit.num_prefix_tokens:].mean(dim=1)
        elif self.vit.global_pool:
            x = x[:, 0]  # class token
        x = self.vit.fc_norm(x)
        x = self.vit.head_drop(x)
        return x if pre_logits else self.vit.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x