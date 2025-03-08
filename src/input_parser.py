# input_parser.py
import argparse
import sys
import json
import yaml
import os

def parse_input_args(input_args):
    # First, create a parser that only parses --reproducibility_statement_file
    initial_parser = argparse.ArgumentParser(description="Train model", add_help=False)
    initial_parser.add_argument(
        '--reproducibility_statement_file',
        default=None,
        help='Path to the reproducibility statement file (JSON or YAML). If provided, parameters are extracted from this file instead of the default values.'
    )

    args, _ = initial_parser.parse_known_args(input_args)

    parser = argparse.ArgumentParser(description="Train model")

    # Define all the arguments
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
        "--lr_start",
        default=1e-6, 
        type=float, 
        help="Learning rate at start of scheduling"
    )
    parser.add_argument(
        "--lr_stop",
        default=1e-5, 
        type=float, 
        help="Learning rate at end of scheduling"
    )
    parser.add_argument(
        "--lr",
        default=1e-3, 
        type=float, 
        help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler_warmup",
        default=0.05, 
        type=float, 
        help="Warmup ratio for scheduler"
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
        default=20, 
        type=int, 
        help="Save model checkpoint every n epochs"
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "vit_tiny_patch16_224",
            "vit_small_patch16_224",
            "vit_base_patch16_224",
            "vit_large_patch16_224"
        ],
        default="vit_tiny_patch16_224",
        type=str,
        help="Insert model name.",
    )
    parser.add_argument(
        "--n_classes", 
        type=int, 
        default=1000,
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
        help="Input image size"
    )
    parser.add_argument(
        "--n_segments",
        default=196,
        type=int,
        help="Number of segments for the tokenizer",
    )
    parser.add_argument(
        "--pretrained_model_path",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--train_from_checkpoint", 
        action="store_true", 
        help="Load model from checkpoint and continue training"
    )
    parser.add_argument(
        '--reproducibility_statement_file',
        default=None,
        help='Path to the reproducibility statement file (JSON or YAML). If provided, parameters are extracted from this file instead of the default values.'
    )
    # New arguments for mixup and cutmix augmentations
    parser.add_argument(
        "--mixup",
        default=0.8,
        type=float,
        help="Mixup alpha value. Set to 0 to disable mixup."
    )
    parser.add_argument(
        "--cutmix",
        default=1.0,
        type=float,
        help="Cutmix alpha value. Set to 0 to disable cutmix."
    )
    parser.add_argument(
        "--cutmix_minmax",
        default=None,
        type=float,
        help="Cutmix minmax value. If provided, activates cutmix with minmax clipping."
    )
    # Additional parameters used in mixup augmentation (if needed)
    parser.add_argument(
        "--mixup_prob",
        default=1.0,
        type=float,
        help="Probability of applying mixup."
    )
    parser.add_argument(
        "--mixup_switch_prob",
        default=0.5,
        type=float,
        help="Probability of switching mixup to cutmix."
    )
    parser.add_argument(
        "--mixup_mode",
        default="batch",
        type=str,
        help="Mode for mixup ('batch' or other supported modes)."
    )
    parser.add_argument(
        "--smoothing",
        default=0.1,
        type=float,
        help="Label smoothing value for soft targets."
    )
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="Number of classes for mixup augmentation."
    )

    # If reproducibility_statement_file is provided, load the args from the file
    if args.reproducibility_statement_file is not None:
        # Load args from the file
        file_ext = os.path.splitext(args.reproducibility_statement_file)[1]
        if file_ext.lower() == '.json':
            with open(args.reproducibility_statement_file, 'r') as f:
                args_from_file = json.load(f)
        elif file_ext.lower() in ['.yaml', '.yml']:
            with open(args.reproducibility_statement_file, 'r') as f:
                args_from_file = yaml.safe_load(f)
        else:
            raise ValueError('Unsupported file extension: {}'.format(file_ext))

        # Set defaults in parser using the parameters from the file
        parser.set_defaults(**args_from_file)

    # Now parse all the arguments (command line args will override defaults)
    args = parser.parse_args(input_args)

    # After parsing the args, save them to a JSON file
    save_args_to_json(args)

    return args

def save_args_to_json(args, filename='reproducibility_statement.json'):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
