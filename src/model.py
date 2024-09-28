import timm
import torch
import torch.nn as nn

from timm.models._manipulate import checkpoint_seq
from feature_extractors.spatial_transformer import AttentionSpatialTransformer
from differentiableWatershed.model import VoronoiPropagation

class differentiableSuperpixelEmbedding(nn.Module):
    def __init__(self, max_segments, n_channels=3, embed_dim=768, max_pixels_per_segment=400):
        super().__init__()
        self.superpixel_tokenizer = VoronoiPropagation(max_segments)
        self.max_segments = max_segments
        self.embed_dim = embed_dim
        self.max_pixels_per_segment = max_pixels_per_segment  # Max number of pixels to sample per segment

        # Linear layer to project raw pixel values to embedding space
        self.feature_projection = nn.Linear(n_channels * max_pixels_per_segment, embed_dim)

    def forward(self, img):
        # Get the superpixel segments from the tokenizer
        _, _, segments = self.superpixel_tokenizer(img)  # segments shape: [batch_size, height, width]
        batch_size, n_channels, height, width = img.shape
        
        # Create tensor to store the features for each segment
        superpixel_features = torch.zeros(batch_size, self.max_segments, n_channels * self.max_pixels_per_segment, device=img.device)

        # Extract raw pixel values per superpixel
        for segment_id in range(self.max_segments):
            segment_mask = (segments == segment_id).unsqueeze(1)  # [batch_size, 1, height, width]
            masked_img = img * segment_mask.float()  # Zero out non-segment pixels
            
            for batch_idx in range(batch_size):
                # Get all pixels in the current segment
                pixels_in_segment = masked_img[batch_idx, :, segment_mask[batch_idx, 0, :, :].bool()]  # [n_channels, num_pixels]
                
                # Flatten and pad/truncate to a fixed size (max_pixels_per_segment)
                num_pixels = pixels_in_segment.shape[1]
                if num_pixels > self.max_pixels_per_segment:
                    # Truncate if there are more pixels than the max allowed
                    pixels_in_segment = pixels_in_segment[:, :self.max_pixels_per_segment]
                else:
                    # Pad with zeros if there are fewer pixels
                    padding = torch.zeros(n_channels, self.max_pixels_per_segment - num_pixels, device=img.device)
                    pixels_in_segment = torch.cat((pixels_in_segment, padding), dim=1)
                
                # Flatten and store in the features tensor
                superpixel_features[batch_idx, segment_id, :] = pixels_in_segment.flatten()

        # Project the raw pixel values to the embedding space
        projected_features = self.feature_projection(superpixel_features)  # [batch_size, max_segments, embed_dim]

        return projected_features
    

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, max_segments, num_classes, num_channels, pretrained=False):
        super().__init__()
        
        
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        self.embed_dim = self.vit.embed_dim
        
        self.patch_embed = differentiableSuperpixelEmbedding(
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