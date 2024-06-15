"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
import torch.nn as nn
import time
import torchvision
from torchvision import transforms
from torch.optim import AdamW
from model import differentiableTokenizerVisionTransformer
from torchinfo import summary
import quixdata
from model_configs import get_config
from scheduler import CosineAnnealingLR_LinearWarmup
import torch.multiprocessing as mp

from utils import train, plot, calculate_warmup_epochs
from train_utils import *
from utils import verify_model_name


def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    verify_model_name(args.model_name)
    model = differentiableTokenizerVisionTransformer(args.model_name, args.n_segments, args.n_classes, args.n_channels)
    summary(model, input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size), depth=4)
    
    optimizer = AdamW(model.parameters(), betas=[args.beta_1, args.beta_2], lr=args.lr, weight_decay=args.weight_decay)
    
    train_dataset, val_dataset = prepare_datasets(args)
    
    train_dataloader = prepare_dataloader(train_dataset, args.batch_size)
    val_dataloader = prepare_dataloader(val_dataset, args.batch_size)
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, rank, args.save_every)
    trainer.train(args.epochs)
    destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data_subfolder_path", default=r"F:\data")
    parser.add_argument("--data_folder_name", default=r"IN1k")
    
    parser.add_argument("--lr_scheduler", default=True, type=bool)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save_every", default=1, type=int)
    
    parser.add_argument("--start_lr", default=0.000001, type=float)
    parser.add_argument("--n_warmup_steps", default=10000, type=int)
    parser.add_argument("--T_max", default=20000, type=int, help="Number of iterations/epochs for cosine annealing scheduler from max to min lr. Typicall 10000, 20000 or 30000.")
    parser.add_argument("--eta_min", default=0.00001, type=float, help="Lowest LR for cosine annealing scheduler. Typically 0.00001 or 0.000001.")
    
    parser.add_argument("--model_name", default = 'vit_base_patch16_224', type=str, help="Insert vit_small, vit_base, vit_large or vit_huge for presets. Enter a custom name if using custom parameters.")
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--mlp_hidden_dim", default=3072, type=int)
    parser.add_argument("--num_attention_heads", default=12, type=int)
    parser.add_argument("--num_transformer_blocks", default=12, type=int)
    
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--n_channels", default=3, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--n_segments", default=50, type=int)
    
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)