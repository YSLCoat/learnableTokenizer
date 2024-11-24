from timm.models._manipulate import checkpoint_seq
from differentiableWatershed.model import VoronoiPropagation

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from differentiableWatershed.model import VoronoiPropagation

class MaskedLinear(nn.Module):
    '''Masked linear layer.

    Linear layer that only computes projections for elements in provided 
    attention mask to save on compute. 
    '''
    def __init__(self, in_feat, out_feat, bias=True, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.act = activation if activation else nn.Identity()
        if bias:
            self.linear.bias.data.mul_(1e-3)
        
    def forward(self, x, amask):
        # x: [N_total, in_feat], amask: [N_total], where N_total = B * N
        masked_indices = amask.nonzero(as_tuple=True)[0]
        if masked_indices.numel() == 0:
            # If no elements are masked, return zeros
            out = torch.zeros_like(x)
            return out
        masked_input = x[masked_indices]  # [num_masked, in_feat]
        masked_output = self.act(self.linear(masked_input))  # [num_masked, out_feat]
        out = torch.zeros(x.shape[0], self.linear.out_features, dtype=x.dtype, device=x.device)
        out[masked_indices] = masked_output
        return out

class MaskedMLP(nn.Module):
    '''Masked MLP module.

    An MLP module that supports masks for training and inference with variable 
    numbers of tokens in each batch.
    '''
    def __init__(self, embed_dim, hid_dim):
        super().__init__()
        self.L1 = MaskedLinear(embed_dim, hid_dim, activation=nn.GELU())
        self.L2 = MaskedLinear(hid_dim, embed_dim)
    
    def forward(self, x, amask):
        x = self.L1(x, amask)
        x = self.L2(x, amask)
        return x

class differentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, n_segments, n_channels=3, embed_dim=768, hid_dim=512):
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

        # Masked MLP for feature aggregation
        self.masked_mlp = MaskedMLP(embed_dim, hid_dim)

    def forward(self, img):
        # Get the superpixel segments
        _, _, segments = self.superpixel_tokenizer(img)  # segments: [B, H, W]
        batch_size, _, height, width = img.shape

        # Process the image with CNN to get feature maps
        features = self.feature_extractor(img)  # [B, embed_dim, Hf, Wf]
        B, C, Hf, Wf = features.shape
        N = Hf * Wf

        # Downsample segments to match feature map size
        segments = F.interpolate(segments.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()  # [B, Hf, Wf]
        segments = segments.view(B, N)  # [B, N]

        # Reshape features
        features = features.view(B, N, C)  # [B, N, C]

        # Flatten batch and pixels for processing
        features_flat = features.view(B * N, C)  # [B*N, C]
        segments_flat = segments.view(B * N)  # [B*N]
        batch_indices = torch.arange(B, device=img.device).unsqueeze(1).expand(B, N).reshape(B * N)  # [B*N]
        global_superpixel_ids = batch_indices * self.n_segments + segments_flat  # [B*N]
        num_superpixels = B * self.n_segments

        # Compute positional embeddings
        pos_embed_table = nn.Embedding(self.n_segments, self.embed_dim).to(img.device)
        pos_embed = pos_embed_table(segments_flat % self.n_segments)  # [B*N, C]

        # Add positional embeddings to features
        features_with_pos = features_flat + pos_embed  # [B*N, C]

        # Apply masked MLP
        mlp_output = self.masked_mlp(features_with_pos, segments_flat >= 0)  # [B*N, C]

        # Group features by superpixel ID and retain all pixel embeddings as tokens
        tokens = torch.zeros((num_superpixels, self.embed_dim), device=img.device)
        tokens[global_superpixel_ids] = mlp_output  # Place pixel embeddings in the correct superpixel group

        # Reshape to match ViT input: [B, n_segments, embed_dim]
        tokens = tokens.view(B, self.n_segments, self.embed_dim)

        return tokens  # Shape: [B, n_segments, embed_dim]
        

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