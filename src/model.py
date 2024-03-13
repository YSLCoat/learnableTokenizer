"""_summary_
Implementation based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
Changelist:
Added scaled_dot_product_attention from pytorch 2.0
"""
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from feature_extractors.spatial_transformer import AttentionSpatialTransformer
from learnable_tokenizers.spixelFCN.models.Spixel_single_layer import SpixelNet

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        A = scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        
        out = rearrange(A, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        if mlp_dim!=4*dim:
            print(f"In original paper the MLP hidden dim is 4 times the embedding dim. MLP dim hidden dim to hidden dim ratio: {mlp_dim/dim}.")

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )
        
        self.superpixel_tokenizer = SpixelNet()
        
        self.feature_extractor = AttentionSpatialTransformer(n_channels=3)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        
        segments = self.superpixel_tokenizer(img)
        segmentation_labels = torch.argmax(segments, dim=1)
        
        batch_segment_features = []
        
        # Iterate over each image in the batch
        for image, segments in zip(img, segmentation_labels):
            segmented_features = []
            #assert 0, (image.shape, segments.shape)
            for segment_label in np.unique(segments.cpu().detach().numpy()):
                segment_mask = segments == segment_label
                segmented_image = image.clone()
                #assert 0, (segmented_image.shape, segment_mask.shape)
                segmented_image[:, ~segment_mask]
                segmented_feature = self.feature_extractor(segmented_image.unsqueeze(0))  # Add batch dimension
                segmented_features.append(segmented_feature)
                
            batch_segment_features.append(segmented_features)
        
        batch_segment_features_tensor = [torch.stack(segmented_features) for segmented_features in batch_segment_features]
        batch_segment_features_flattened = [segmented_features.view(segmented_features.size(0), -1) for segmented_features in batch_segment_features_tensor]
        
        max_num_segments = max(len(segmented_features) for segmented_features in batch_segment_features_flattened)

        # Pad or truncate the segmented features to ensure they all have the same length
        for i in range(len(batch_segment_features_flattened)):
            num_segments = batch_segment_features_flattened[i].size(0)
            if num_segments < max_num_segments:
                # If the number of segments is less than the maximum, pad with zeros
                padding_size = max_num_segments - num_segments
                padding = torch.zeros(padding_size, batch_segment_features_flattened[i].size(1))
                batch_segment_features_flattened[i] = torch.cat([batch_segment_features_flattened[i], padding], dim=0)
            elif num_segments > max_num_segments:
                # If the number of segments is greater than the maximum, truncate
                batch_segment_features_flattened[i] = batch_segment_features_flattened[i][:max_num_segments]

        batch_segment_features_flattened = [segmented_features.view(-1) for segmented_features in batch_segment_features_flattened]
        batch_segment_features_flattened = torch.stack(batch_segment_features_flattened)
        
        #x = self.to_patch_embedding(img)
        b, n, _ = batch_segment_features_flattened.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, batch_segment_features_flattened), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
if __name__ == "__main__":

    def test_ViT():
        model = ViT(
            image_size=256,
            patch_size = 32,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            channels=3,
            num_classes=1000,
        )

        img = torch.randn(1, 3, 256, 256)

        preds = model(img)
        assert preds.shape == (1, 1000), 'incorrect logits outputted'
        print(f"Test passed. Logits shape [batch_size, n_classes] :{preds.shape}")
        
    test_ViT()