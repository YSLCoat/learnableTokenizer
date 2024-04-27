import timm
import torch
import torch.nn as nn

from timm.models._manipulate import checkpoint_seq



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 100
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes).to(device)

# Subclassing the Vision Transformer
class CustomVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the patch embedding layer
        self.patch_embed = CustomPatchEmbedding(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=self.embed_dim
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x