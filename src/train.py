import torch
import sys

import torch.nn as nn
import torch.multiprocessing as mp

from torchinfo import summary
from torch.distributed import destroy_process_group

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler  # or CosineLRScheduler, etc.

from model import DifferentiableSuperpixelTokenizerViT
from train_utils_vit import ddp_setup, Trainer, prepare_datasets, prepare_dataloader
from utils import get_available_gpus
from input_parser import parse_input_args


def main(rank, world_size, args):
    args = parse_input_args(args)

    ddp_setup(rank, world_size)

    model = DifferentiableSuperpixelTokenizerViT(
        args.model_name, args.n_segments, args.n_classes, args.n_channels, args.superpixel_algorithm, args.pretrained
    )

    if rank == 0:
        summary(
            model,
            input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size),
            depth=4
        )

    optimizer = create_optimizer_v2(
        model,
        opt='adamw',
        lr=args.base_lr,  # The base LR; it will be scaled if you pass scale_lr in create_scheduler
        weight_decay=args.weight_decay,
        betas=(args.beta_1, args.beta_2),
        layer_decay=0.65
    )

    train_dataset, val_dataset = prepare_datasets(args)
    train_dataloader = prepare_dataloader(train_dataset, args.batch_size)
    val_dataloader = prepare_dataloader(val_dataset, args.batch_size)

    # check if mixup or cutmix is active
    mixup_active = (
        args.mixup > 0 or args.cutmix > 0
        or (hasattr(args, "cutmix_minmax") and args.cutmix_minmax is not None)
    )
    mixup_fn = None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup,
    #         cutmix_alpha=args.cutmix,
    #         cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob,
    #         switch_prob=args.mixup_switch_prob,
    #         mode=args.mixup_mode,
    #         label_smoothing=args.smoothing,
    #         num_classes=args.n_classes  # or args.nb_classes if you prefer
    #     )

    # If using mixup, we generally use SoftTargetCrossEntropy to handle soft labels
    if mixup_active:
        pass
        # loss_function = SoftTargetCrossEntropy()
    else:
        label_smoothing = getattr(args, "label_smoothing", 0.0)
        loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scheduler, _ = create_scheduler(args, optimizer)

    trainer = Trainer(
        args=args,
        model=model,
        train_data=train_dataloader,
        val_data=val_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        gpu_id=rank,
        save_every=args.save_every,
        mixup_function=mixup_fn,
        scheduler=scheduler
    )

    trainer.train(args.epochs)
    destroy_process_group()


if __name__ == '__main__':
    get_available_gpus()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, sys.argv[1:]), nprocs=world_size)
