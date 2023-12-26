import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import pair

class DepthwiseTokenizer(nn.Module):
    def __init__(self, patch_size, num_patches, embed_dim, channels=3):
        super().__init__()
        # Store dimension sizes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.vec_dim = patch_size**2 * channels

        # Add learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1,1,self.embed_dim))

        # Define modules
        self.conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size)
        
        # Add learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
    def forward(self, x):
        p = self.patch_size
        b, c, h, w = x.shape # Assume image is channels first
        assert c == self.channels, f"Input channels does not match, {c}!={self.channels}!"

        # Pad image to match patch size
        x = F.pad(x, (0, w%p, 0, h%p), mode='constant', value=0)

        # Use depthwise convolution to divide image into patches and do the projection
        x = self.conv(x)
        
        # Flatten grid of patches and transpose last two dimensions
        x = x.view(b, -1, self.embed_dim)
        
        b, n, _ = x.shape
        
        # Expand and concatenate class token
        cls_token = self.cls_token.expand(b, -1, -1)
        
        x = torch.cat([cls_token, x], dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
    
        return x
    

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
    
    
class MSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f"embed_dim must be divisible by num_heads"

        # Projections
        self.U_qkv = nn.ModuleList([nn.Linear(self.embed_dim, 3*self.head_dim) for _ in range(num_heads)])
        self.U_msa = nn.Linear(self.num_heads * self.head_dim, self.embed_dim)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        
        # Empty tensor to store the results
        out = torch.empty((*z.shape[:-1], 0), dtype=z.dtype, layout=z.layout, device=z.device)

        for head in range(self.num_heads):

            # Project input and split into query, key and value
            q, k, v = self.U_qkv[head](z).chunk(3, dim=-1)
            
            A = scaled_dot_product_attention(q, k, v)
            out = torch.cat((out, A), dim=-1)
            
        # Final projection of the concatenation of the heads
        msa = self.U_msa(out)
        
        return msa
    

class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    def __init__(self,
                 embedding_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), 
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer.."
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, mlp_dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim

        self.layernorm1 = nn.LayerNorm(embed_dim)
        # Initialize the attention block
        self.msa = MSA(embed_dim, num_heads)
        # Initialize MLPBlock
        self.mlp = MLPBlock(embed_dim, self.mlp_hidden_dim, mlp_dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.layernorm1(x)
        x = x + self.msa(x)
        return x + self.mlp(self.layernorm2(x))

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads, num_blocks, mlp_hidden_dim, channels=3, num_classes=None):
        super().__init__()

        # Store dimension sizes
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.channels = channels
        self.vec_dim = patch_size**2 * channels
        self.num_heads = num_heads

        # Create modules
        self.tokenize = DepthwiseTokenizer(patch_size, num_patches, embed_dim, channels=channels)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, mlp_hidden_dim) for _ in range(num_blocks)])
        self.layernorm = nn.LayerNorm(embed_dim)
        
        # MLP head for classification
        if num_classes != None:
            self.num_classes = num_classes
            self.mlp_head = nn.Sequential(
                nn.Linear(self.embed_dim, num_classes),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        # Tokenize
        x = self.tokenize(x)

        # Encoder
        for block in self.blocks:
            x = block(x)

        # Extract only classification token
        y = self.layernorm(x[:,0])

        # MLP head for classification
        if self.num_classes != None:
            y = self.mlp_head(y)

        return y
    
if __name__ == "__main__":
    seed = 42

    def test_DepthwiseTokenizer():
        torch.manual_seed(seed)
        patch_size = 4
        embed_dim = 10
        image_size = 28
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        depth_tokenizer = DepthwiseTokenizer(patch_size, num_patches, embed_dim)

        x = torch.rand(1, 3, image_size, image_size)
        
        depth_x = depth_tokenizer(x)
        expected_shape = (1, (28//patch_size)*(28//patch_size) + 1, embed_dim)
        
        assert depth_x.shape == expected_shape, f"DepthwiseTokenizer output has shape {depth_x.shape}, but should be {expected_shape}"

    test_DepthwiseTokenizer()
    
    def test_MSA():
        embed_dim = 10
        num_heads = 2

        torch.manual_seed(seed)
        msa = MSA(embed_dim, num_heads)
        
        tokens = torch.rand(1, 32, embed_dim)
        out_corr = torch.rand(1, 32, embed_dim)
        
        out = msa(tokens)

        assert out.shape == out_corr.shape, f"MSA output has shape {out.shape}, but should be {out_corr.shape}."
        
    test_MSA()

    def test_ViT():
        model = VisionTransformer(
            image_size=256,
            patch_size = 32,
            embed_dim = 1024,
            num_blocks = 6,
            num_heads = 16,
            mlp_hidden_dim = 2048,
            channels=3,
            num_classes=1000,
        )

        img = torch.randn(1, 3, 256, 256)

        preds = model(img)
        assert preds.shape == (1, 1000), 'correct logits outputted'
        
    test_ViT()