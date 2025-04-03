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
        args.model_name, args.n_segments, args.n_classes, args.n_channels, args.superpixel_algorithm
    )

    if rank == 0:
        summary(
            model,
            input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size),
            depth=4
        )

    # We will rely on timm's create_optimizer_v2 for layer-wise LR decay, AdamW, etc.
    # 'layer_decay=0.65' is a common choice for large ViTs; adjust as needed.
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
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.n_classes  # or args.nb_classes if you prefer
        )

    # If using mixup, we generally use SoftTargetCrossEntropy to handle soft labels
    if mixup_active:
        loss_function = SoftTargetCrossEntropy()
    else:
        label_smoothing = getattr(args, "label_smoothing", 0.0)
        loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # -------------------------------------------------------------------------
    # (Optional) Scheduler
    # -------------------------------------------------------------------------
    # If you'd like a warmup + cosine schedule, we can create it via timm's create_scheduler
    # Example: timm passes a 'steps_per_epoch' argument to properly do step-based scheduling
    # 'decay_epochs' -> when to end the cycle, 'warmup_epochs' -> # of warmup epochs, etc.
    #
    # Additional references:
    #   - timm.scheduler.CosineLRScheduler
    #   - timm.scheduler.create_scheduler
    #
    # We'll do a quick example here:
    #
    steps_per_epoch = len(train_dataloader)
    scheduler, _ = create_scheduler(args, optimizer)
    # create_scheduler expects that you have certain attributes in args, like:
    #   args.sched, args.epochs, args.decay_epochs, args.warmup_epochs,
    #   args.cooldown_epochs, etc. 
    # Example:
    #   args.sched = 'cosine'
    #   args.epochs = 100
    #   args.decay_epochs = 100
    #   args.warmup_epochs = 5
    # If you do not have these, you'll need to specify them or default them in parse_input_args.

    # Alternatively, if you prefer a simpler approach:
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)


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
        # If your Trainer constructor takes scheduler, pass it here, e.g.:
        # scheduler=scheduler
    )

    trainer.train(args.epochs)
    destroy_process_group()


if __name__ == '__main__':
    get_available_gpus()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, sys.argv[1:]), nprocs=world_size)
