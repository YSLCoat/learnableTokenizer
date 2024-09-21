# arg_parse.py
import argparse
import sys


def parse_input_args(input_args):
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument(
        "--data_subfolder_path", 
        default=r"F:\data", 
        help="Path to the data subfolder"
    )
    parser.add_argument(
        "--data_folder_name", 
        default=r"IN1k", 
        help="Name of the data folder"
    )
    parser.add_argument(
        "--lr_scheduler",
        default=True,
        type=bool,
        help="Whether to use a learning rate scheduler",
    )
    parser.add_argument(
        "--lr",
        default=1e-4, 
        type=float, 
        help="Learning rate"
    )
    parser.add_argument(
        "--beta_1", 
        default=0.9, 
        type=float, 
        help="Beta 1 parameter for AdamW optimizer"
    )
    parser.add_argument(
        "--beta_2",
        default=0.999,
        type=float,
        help="Beta 2 parameter for AdamW optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--batch_size", 
        default=64, 
        type=int, 
        help="Batch size"
    )
    parser.add_argument(
        "--epochs", 
        default=500, 
        type=int, 
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--save_every", 
        default=1, 
        type=int, 
        help="Save model checkpoint every n epochs"
    )
    parser.add_argument(
        "--model_name",
        default="vit_base_patch16_224",
        type=str,
        help="Insert vit_small, vit_base, vit_large or vit_huge for presets. Enter a custom name if using custom parameters.",
    )
    parser.add_argument(
        "--embed_dim", 
        default=768, 
        type=int, 
        help="Embedding dimension size"
    )
    parser.add_argument(
        "--mlp_hidden_dim", 
        default=3072, 
        type=int, 
        help="MLP hidden dimension size"
    )
    parser.add_argument(
        "--num_attention_heads",
        default=12,
        type=int,
        help="Number of attention heads in the transformer",
    )
    parser.add_argument(
        "--num_transformer_blocks",
        default=12,
        type=int,
        help="Number of transformer blocks",
    )
    parser.add_argument(
        "--n_classes", 
        type=int, 
        required=True, 
        help="Number of output classes"
    )
    parser.add_argument(
        "--n_channels", 
        default=3, 
        type=int, 
        help="Number of input channels"
    )
    parser.add_argument(
        "--img_size", 
        default=224, 
        type=int, 
        help="Input image size")
    parser.add_argument(
        "--n_segments",
        default=50,
        type=int,
        help="Number of segments for the tokenizer",
    )

    # If input_args is None, use sys.argv[1:]
    if input_args is None:
        input_args = sys.argv[1:]

    return parser.parse_args(input_args)
