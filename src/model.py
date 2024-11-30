import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import timm
from differentiableWatershed.model import VoronoiPropagation

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

class GraphBasedEmbedder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.gcn = GCNConv(in_channels, embed_dim, add_self_loops=True)  # One layer GCN

    def forward(self, features, edge_index, batch):
        """
        features: [total_nodes, in_channels] - Features of all nodes (superpixels)
        edge_index: [2, num_edges] - Edge connections between nodes
        batch: [total_nodes] - Batch index for each node
        """
        return self.gcn(features, edge_index)  # Returns [total_nodes, embed_dim]


class DifferentiableSuperpixelTokenizer(nn.Module):
    def __init__(self, n_channels, n_segments, embed_dim=768):
        super().__init__()
        self.n_channels = n_channels
        self.n_segments = n_segments
        self.embed_dim = embed_dim
        self.superpixel_tokenizer = VoronoiPropagation(n_segments)

        # Linear projection to node features (patch embeddings)
        self.node_projector = nn.Linear(n_channels, embed_dim)

        # Graph-based embedding
        self.graph_embedder = GraphBasedEmbedder(embed_dim, embed_dim)

    def forward(self, img):
        B, C, H, W = img.size()
        device = img.device

        # Step 1: Generate superpixel segments
        _, _, segments = self.superpixel_tokenizer(img)  # segments: [B, H, W]

        # Step 2: Flatten image and segments
        img_flat = img.view(B, C, -1)  # [B, C, H*W]
        segments_flat = segments.view(B, -1)  # [B, H*W]

        # Step 3: Compute features for each superpixel
        segment_ids = torch.arange(self.n_segments, device=device)  # All possible segment IDs
        segment_masks = (segments_flat.unsqueeze(1) == segment_ids.unsqueeze(0).unsqueeze(2))  # [B, n_segments, H*W]
        
        # Calculate pixel counts per superpixel for normalization
        segment_pixel_counts = segment_masks.sum(dim=-1, keepdim=True).clamp_min(1)  # [B, n_segments, 1]

        # Apply masks to extract features
        masked_features = segment_masks.unsqueeze(2) * img_flat.unsqueeze(1)  # [B, n_segments, C, H*W]
        segment_features = masked_features.sum(dim=-1) / segment_pixel_counts  # [B, n_segments, C]

        # Step 4: Project features to embedding space
        projected_features = self.node_projector(segment_features)  # [B, n_segments, embed_dim]

        # Step 5: Create edges (fully connected graph)
        edge_index_list = []
        batch_indices = []
        for b in range(B):
            edge_index = torch.combinations(torch.arange(self.n_segments, device=device), r=2).T
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make bidirectional
            edge_index_list.append(edge_index)
            batch_indices.append(torch.full((self.n_segments,), b, dtype=torch.long, device=device))

        # Combine edge indices for the batch
        edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_edges]
        batch = torch.cat(batch_indices, dim=0)  # [total_nodes]

        # Flatten projected features across the batch
        embeddings = projected_features.view(-1, self.embed_dim)  # [total_nodes, embed_dim]

        # Step 6: Graph-based embedding
        graph_embeddings = self.graph_embedder(embeddings, edge_index, batch)  # [total_nodes, embed_dim]

        # Step 7: Reshape into [B, n_segments, embed_dim]
        output = graph_embeddings.view(B, self.n_segments, self.embed_dim)

        return output
    
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
