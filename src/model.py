import timm
import torch
import torch.nn as nn

from timm.models._manipulate import checkpoint_seq
from feature_extractors.spatial_transformer import AttentionSpatialTransformer
from learnable_tokenizers.spixelFCN.models.Spixel_single_layer import SpixelNet



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class differentiableSuperpixelEmbedding(nn.Module):
    def __init__(self, max_segments, n_channels=3, embed_dim=768):
        super().__init__()
        self.superpixel_tokenizer = SpixelNet(max_segments).to('cuda')
        self.feature_extractor = AttentionSpatialTransformer(embed_dim, n_channels=n_channels).to('cuda')
        self.max_segments = max_segments  # This should be the maximum expected number of segments

    def forward(self, img):
        segments = self.superpixel_tokenizer(img)
        _, segmentation_labels = torch.max(segments, dim=1)
        
        all_segmented_images = []
        batch_segment_indices = []

        for batch_idx, (image, segments) in enumerate(zip(img, segmentation_labels)):
            unique_segments = torch.unique(segments)
            for label in unique_segments:
                segment_mask = (segments == label).unsqueeze(0)
                segmented_image = image * segment_mask
                all_segmented_images.append(segmented_image)
                batch_segment_indices.append((batch_idx, label))
        
        # Convert list to tensor for batch processing
        all_segmented_images = torch.stack(all_segmented_images)
        
        # Process all segments in one batch
        all_segmented_features = self.feature_extractor(all_segmented_images)
        flattened_features = all_segmented_features.view(all_segmented_features.size(0), -1)
        
        # Initialize tensor to hold batch features, with padding to max_segments
        batch_segment_features = torch.zeros((img.size(0), self.max_segments, flattened_features.size(1)), device=img.device)

        # Assign the features back to the appropriate positions in the batch
        segment_counts = torch.zeros(img.size(0), dtype=torch.long, device=img.device)
        for (batch_idx, _), feature in zip(batch_segment_indices, flattened_features):
            segment_idx = segment_counts[batch_idx]
            if segment_idx < self.max_segments:
                batch_segment_features[batch_idx, segment_idx, :] = feature
                segment_counts[batch_idx] += 1

        return batch_segment_features

class differentiableTokenizerVisionTransformer(nn.Module):
    def __init__(self, model_name, pretrained, max_segments, num_classes, num_channels):
        super().__init__()
        
        
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes).to(device)
        
        self.embed_dim = self.vit.embed_dim
        
        self.patch_embed = differentiableSuperpixelEmbedding(
            max_segments=max_segments,
            n_channels=num_channels,
            embed_dim=self.embed_dim
        ).to(device)

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