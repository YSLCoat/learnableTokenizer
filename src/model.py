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
        
        self.superpixel_tokenizer = SpixelNet().to('cuda')
        self.feature_extractor = AttentionSpatialTransformer(embed_dim, n_channels=n_channels).to('cuda')
        
    def forward(self, img):
        segments = self.superpixel_tokenizer(img)
        _, segmentation_labels = torch.max(segments, dim=1)
        
        batch_segment_features = []
        for image, segments in zip(img, segmentation_labels):
            segmented_features = []
            for segment_label in torch.unique(segments):
                segment_mask = segments == segment_label
                # Apply the mask to the image and retain only relevant parts of the image
                segmented_image = image * segment_mask.unsqueeze(0)  # Ensure the mask has the same dimension as the image
                segmented_feature = self.feature_extractor(segmented_image.unsqueeze(0))  # Add batch dimension
                # Flatten the output feature to a vector
                flattened_feature = segmented_feature.view(-1)
                segmented_features.append(flattened_feature)
            
            batch_segment_features.append(torch.stack(segmented_features))

        max_num_segments = max(len(features) for features in batch_segment_features)
        # Pad the segmented features tensor to ensure they all have the same length
        batch_segment_features_padded = []
        for features in batch_segment_features:
            if len(features) < max_num_segments:
                # Calculate padding size for the last dimension (flattened feature dimension)
                padding = torch.zeros(max_num_segments - len(features), features.size(1))
                features_padded = torch.cat([features, padding], dim=0)
            else:
                features_padded = features[:max_num_segments]
            batch_segment_features_padded.append(features_padded)
        
        batch_segment_features_tensor = torch.stack(batch_segment_features_padded)
        
        return batch_segment_features_tensor

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
        self.vit.pos_embed = nn.Parameter(torch.randn(1, max_segments, self.embed_dim) * .02)
        self.vit.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)) 
        
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        b, n, d = x.shape
        cls_tokens = self.vit.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vit.pos_embed[:, :(n + 1)] # n is the number of segments
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