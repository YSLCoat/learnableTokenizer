import torch
import sys

from torch.optim import AdamW
from model import DifferentiableSuperpixelTokenizer
from torchinfo import summary
import torch.multiprocessing as mp

from train_utils import *
from utils import get_available_gpus
from input_parser import parse_input_args
from torch.distributed import destroy_process_group


def main(rank, world_size, args):
    args = parse_input_args(args)
    ddp_setup(rank, world_size)
    
    model = DifferentiableSuperpixelTokenizer(
        max_segments=args.n_segments, 
        n_channels=args.n_channels,
        embed_dim=192,
        use_positional_embeddings=False,
        reconstruction=True,
        superpixel_algorithm=args.superpixel_algorithm,
    )
    
    # if rank==0:
    #     summary(model, input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size), depth=4)
    
    optimizer = AdamW(
        model.parameters(),
        betas=[args.beta_1, args.beta_2],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
     
    train_dataset, val_dataset = prepare_datasets(args)
    
    train_dataloader = prepare_dataloader(train_dataset, args.batch_size)
    val_dataloader = prepare_dataloader(val_dataset, args.batch_size)
    
    loss_function = nn.MSELoss()
            
    trainer = Trainer(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_function,
        rank,
        args.save_every
    )
    
    trainer.train(args.epochs)
    destroy_process_group()

if __name__ == '__main__':
    get_available_gpus()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, sys.argv[1:]), nprocs=world_size)