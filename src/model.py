import timm
import torch
import torch.nn as nn

from timm.models._manipulate import checkpoint_seq
from feature_extractors.spatial_transformer import AttentionSpatialTransformer
from learnable_tokenizers.spixelFCN.models.Spixel_single_layer import SpixelNet
from differentiableWatershed.model import VoronoiPropagation

class differentiableSuperpixelEmbedding(nn.Module):
    def __init__(self, max_segments, n_channels=3, embed_dim=768):
        super().__init__()
        self.superpixel_tokenizer = VoronoiPropagation(max_segments)
        self.feature_extractor = AttentionSpatialTransformer(embed_dim, n_channels=n_channels)
        self.max_segments = max_segments  # This should be the maximum expected number of segments

    def forward(self, img):
        # Get the superpixel segments from the tokenizer
        _, _, segments = self.superpixel_tokenizer(img)  # segments shape: [batch_size, height, width]
        
        batch_size, n_channels, height, width = img.shape
        
        # Flatten the spatial dimensions for efficient indexing
        img_flat = img.view(batch_size, n_channels, -1)  # [batch_size, n_channels, height * width]
        segments_flat = segments.view(batch_size, -1)  # [batch_size, height * width]
        
        # Create a tensor to accumulate the masked regions for each segment
        segment_features = torch.zeros(batch_size, self.max_segments, n_channels, height, width, device=img.device)
        
        # Loop over each segment ID and use scatter_add to aggregate the pixel values
        for segment_id in range(self.max_segments):
            # Create a mask for the current segment across all images in the batch
            segment_mask = (segments_flat == segment_id).unsqueeze(1)  # [batch_size, 1, height * width]
            
            # Scatter the pixel values for each segment into the output tensor
            masked_img_flat = img_flat * segment_mask.float()  # Zero out pixels not in the segment
            segment_features[:, segment_id, :, :, :] += masked_img_flat.view(batch_size, n_channels, height, width)

        # Flatten segments and batch for processing in one go
        segment_features_flat = segment_features.view(batch_size * self.max_segments, n_channels, height, width)
        
        # Apply feature extractor to each segment
        extracted_features = self.feature_extractor(segment_features_flat)
        
        # Reshape back to original batch and segment size
        extracted_features = extracted_features.view(batch_size, self.max_segments, -1)  # [batch_size, max_segments, feature_dim]
        
        return extracted_features

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