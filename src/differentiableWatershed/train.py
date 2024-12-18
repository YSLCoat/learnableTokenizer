import torch
from .data_utils import prepare_dataloader
from .trainingutils import ddp_setup, Trainer
from .input_parser import parse_input_args
from .model import DifferentiableVoronoiPropagation
from torch.optim import AdamW

from torch.distributed import destroy_process_group
import torch.multiprocessing as mp

import sys


def main(args, rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    model = DifferentiableVoronoiPropagation()
    train_loader, val_loader = prepare_dataloader(args.data_path, args.batch_size)
    
    optimizer = AdamW(
        model.parameters(),
        betas=[args.beta_1, args.beta_2],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    args = parse_input_args(sys.argv[1:])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(args, world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)