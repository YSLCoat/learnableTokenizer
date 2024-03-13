def ViT_Tiny():
    return dict(
        num_transformer_blocks=12,
        embed_dim=192,
        mlp_hidden_dim=768,
        num_attention_heads=3
    )

def ViT_Small():
    return dict(
        num_transformer_blocks=12,
        embed_dim=384,
        mlp_hidden_dim=1536,
        num_attention_heads=6
    )

def ViT_Base():
    return dict(
        num_transformer_blocks=12,
        embed_dim=768,
        mlp_hidden_dim=3072,
        num_attention_heads=12
    )

def ViT_Large():
    return dict(
        num_transformer_blocks=24,
        embed_dim=1024,
        mlp_hidden_dim=4096,
        num_attention_heads=16
    )

def ViT_Huge():
    return dict(
        num_transformer_blocks=32,
        embed_dim=1280,
        mlp_hidden_dim=5120,
        num_attention_heads=16
    )

def get_config(model_name):
    config_functions = {
        "vit_tiny": ViT_Small,
        "vit_small": ViT_Small,
        "vit_base": ViT_Base,
        "vit_large": ViT_Large,
        "vit_huge": ViT_Huge
    }
    
    if model_name.lower() in config_functions:
        return config_functions[model_name.lower()]
    else:
        print(f"Warning: Model name '{model_name}' not in list of pre-defined configurations. If custom parameters are defined, these will be used. If not, default parameters will be used (ViT Base).")
        return None