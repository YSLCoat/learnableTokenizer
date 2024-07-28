import timm
import torch
import torch.nn as nn

from timm.models._manipulate import checkpoint_seq
from feature_extractors.spatial_transformer import AttentionSpatialTransformer
from learnable_tokenizers.spixelFCN.models.Spixel_single_layer import SpixelNet



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class differentiableSuperpixelEmbedding(nn.Module):
    def __init__(self, n_segments, n_channels, embed_dim, device):
        super().__init__()
        self.superpixel_tokenizer = SpixelNet(n_segments).to(device)
        self.feature_extractor = AttentionSpatialTransformer(embed_dim, n_channels=n_channels).to(device)
        self.n_segments = n_segments  # This should be the maximum expected number of segments

    def forward(self, img):
        segments = self.superpixel_tokenizer(img)
        _, segmentation_labels = torch.max(segments, dim=1)
        
        batch_size, _, height, width = img.size()
        n_segments = self.n_segments
        
        # Initialize a tensor to store segmented images
        all_segmented_images = torch.zeros((batch_size, n_segments, *img.shape[1:]), device=img.device)
        
        # Get unique segment labels and their counts for each image in the batch
        unique_segments_per_batch = [torch.unique(segmentation_labels[batch_idx]) for batch_idx in range(batch_size)]
        max_unique_segments = max([len(unique_segments) for unique_segments in unique_segments_per_batch])
        
        # Stack all segment labels to a fixed size tensor for vectorized operations
        all_unique_segments = torch.zeros((batch_size, n_segments), dtype=torch.long, device=img.device)
        for batch_idx, unique_segments in enumerate(unique_segments_per_batch):
            all_unique_segments[batch_idx, :len(unique_segments)] = unique_segments
        
        # Create masks for all segments
        segment_masks = (segmentation_labels.unsqueeze(1) == all_unique_segments.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width))
        
        # Ensure we do not exceed n_segments
        segment_masks = segment_masks[:, :n_segments, :, :]
        
        # Expand masks to match the input image channels
        segment_masks = segment_masks.unsqueeze(2).expand(-1, -1, img.size(1), -1, -1)
        
        # Apply masks to the images
        all_segmented_images = (img.unsqueeze(1) * segment_masks).view(-1, *img.shape[1:])
        
        # Process all segments in one batch
        all_segmented_features = self.feature_extractor(all_segmented_images)
        flattened_features = all_segmented_features.view(all_segmented_features.size(0), -1)
        
        # Reshape the features to the original batch and segment structure
        batch_segment_features = flattened_features.view(batch_size, n_segments, -1)
        
        return batch_segment_features

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, pretrained, n_segments, num_classes, num_channels, device):
        super().__init__()
        
        
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes).to(device)
        
        self.embed_dim = self.vit.embed_dim
        
        self.patch_embed = differentiableSuperpixelEmbedding(
            n_segments=n_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim,
            device=device
        ).to(device)

        self.vit.patch_embed = self.patch_embed
        self.vit.pos_embed = nn.Parameter(torch.randn(1, n_segments + 1, self.embed_dim) * .02)
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