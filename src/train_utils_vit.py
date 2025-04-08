import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
torchvision.disable_beta_transforms_warning()

import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torch.nn as nn
from torchvision.transforms.v2 import RandAugment
from data_utils import mixup_augmentation
from timm.loss import SoftTargetCrossEntropy

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import os
from tqdm import tqdm
import time
from datetime import timedelta

import quixdata
from utils import load_model_from_state_dict, visualize_segmentation
from metrics import explained_variance_batch


def ddp_setup(rank, world_size):
    """
    Initializes the distributed training environment.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    # pick any free port, e.g. "12400"
    os.environ["MASTER_PORT"] = "12401"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        args,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function,
        gpu_id: int,
        save_every: int,
        mixup_function,
        scheduler=None,  # If you want to add a scheduler, pass it here
    ) -> None:
        """
        Args:
            args: Namespace with training hyperparameters
            model: Your ViT-based model
            train_data: Dataloader for training
            val_data: Dataloader for validation
            optimizer: Typically AdamW
            loss_function: SoftTargetCrossEntropy (if mixup is active) else CrossEntropy
            gpu_id: GPU device index for this process
            save_every: Save checkpoint every N epochs
            mixup_function: A function or None for applying mixup/cutmix
            # scheduler: (Optional) LR scheduler, e.g. warmup + cosine
        """
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.save_every = save_every
        self.mixup_augmentation = mixup_function
        self.scheduler = scheduler  # uncomment if using a scheduler
        self.verbose_training = True

        # Load from a checkpoint if specified
        if args.train_from_checkpoint:
            self.model = load_model_from_state_dict(self.model, args.pretrained_model_path)

        # Wrap in DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=False)

        # For logging
        self.results = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        self.args = args

    def _run_batch(self, source, targets, train=False):
        """
        Runs a single batch forward/backward pass.
        """
        if train:
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass
            preds, gradient_map, segments = self.model(source)

            # converted targets to soft labels. If not, do it here:
            if self.mixup_augmentation is not None:
                # In your code you do a one-hot conversion. That’s good if using
                # standard CrossEntropy for multi-class classification.
                targets_for_loss = F.one_hot(targets, num_classes=self.args.n_classes).float()
            else:
                targets_for_loss = targets

            # Compute loss
            loss = self.loss_function(preds, targets_for_loss)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # if self.scheduler is not None:
            #     self.scheduler.step()

            return loss.item(), preds, segments, gradient_map

        else:
            self.model.eval()
            with torch.no_grad():
                preds, gradient_map, segments = self.model(source)
                # For validation, we typically assume integer labels and use cross-entropy
                # or we can skip if you prefer to keep consistent with mixup
                loss = F.cross_entropy(preds, targets)
                return loss.item(), preds, segments, gradient_map

    def _run_epoch(self, epoch):
        """
        Train for one epoch, then validate.
        """
        self.train_data.sampler.set_epoch(epoch)

        train_loss = 0.0
        train_ev_total = 0.0
        num_train_batches = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (source, targets) in enumerate(tqdm(self.train_data, desc=f"Train Epoch {epoch+1}")):
            # Move data to GPU
            source = source.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)

            loss, preds, segments, gradient_map = self._run_batch(source, targets, train=True)
            train_loss += loss

            # Classification accuracy
            pred_labels = preds.argmax(dim=1)
            train_correct += (pred_labels == targets).sum().item()
            train_total += targets.size(0)

            # Compute explained variance
            mean = torch.tensor([0.485, 0.456, 0.406], device=source.device).view(1, -1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=source.device).view(1, -1, 1, 1)
            unnormalized_batch = (source * std) + mean

            batch_ev_scores = explained_variance_batch(unnormalized_batch, segments)
            batch_ev = sum(batch_ev_scores) / len(batch_ev_scores)
            train_ev_total += batch_ev
            num_train_batches += 1

            # Print intermediate logs
            if self.verbose_training and (batch_idx % 100 == 0):
                print(f"Epoch: {epoch+1} | Batch: {batch_idx+1} | Loss: {loss:.4f} | EV: {batch_ev:.4f}")

            if batch_idx % 1000 == 0:
                sample_image = unnormalized_batch[0]
                sample_gradient = gradient_map[0]
                sample_segments = segments[0]
                output_folder = "visualizations"
                filename = f"epoch_{epoch+1}_batch_{batch_idx+1}.png"
                visualize_segmentation(sample_image, sample_gradient, sample_segments, output_folder, filename)

        train_loss /= len(self.train_data)
        train_ev = train_ev_total / num_train_batches
        train_acc = train_correct / train_total


        val_loss = 0.0
        val_ev_total = 0.0
        num_val_batches = 0
        val_correct = 0
        val_total = 0

        for batch_idx, (source, targets) in enumerate(tqdm(self.val_data, desc=f"Val Epoch {epoch+1}")):
            source = source.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)

            loss, preds, segments, _ = self._run_batch(source, targets, train=False)
            val_loss += loss

            # Accuracy
            pred_labels = preds.argmax(dim=1)
            val_correct += (pred_labels == targets).sum().item()
            val_total += targets.size(0)

            # Explained variance
            mean = torch.tensor([0.485, 0.456, 0.406], device=source.device).view(1, -1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=source.device).view(1, -1, 1, 1)
            unnormalized_batch = (source * std) + mean

            batch_ev_scores = explained_variance_batch(unnormalized_batch, segments)
            batch_ev = sum(batch_ev_scores) / len(batch_ev_scores)
            val_ev_total += batch_ev
            num_val_batches += 1

            if self.verbose_training and (batch_idx % 100 == 0):
                print(f"Val Epoch: {epoch+1} | Batch: {batch_idx+1} | Loss: {loss:.4f} | EV: {batch_ev:.4f}")

        val_loss /= len(self.val_data)
        val_ev = val_ev_total / num_val_batches
        val_acc = val_correct / val_total

        if self.scheduler is not None:
            self.scheduler.step(epoch)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | train_ev: {train_ev:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_ev: {val_ev:.4f} | val_acc: {val_acc:.4f}"
        )

    def _save_checkpoint(self, epoch):
        """
        Saves the model state dict as a checkpoint.
        """
        ckp = self.model.module.state_dict()
        torch.save(ckp, self.args.model_save_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {self.args.model_save_path}")

    def train(self, max_epochs: int):
        """
        Main training loop over epochs.
        """
        for epoch in range(max_epochs):
            start = time.time()
            self._run_epoch(epoch)
            end = time.time()

            # Print initial epoch timing and estimate total
            if epoch == 0:
                elapsed_time_seconds = end - start
                elapsed_time_formatted = str(timedelta(seconds=int(elapsed_time_seconds)))
                total_est = str(timedelta(seconds=int(elapsed_time_seconds * max_epochs)))
                print(f"First epoch took: {elapsed_time_formatted}. Estimated training time: {total_est}.")

            # Save checkpoint periodically
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)



def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
    Creates a distributed DataLoader with a DistributedSampler.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=4,
    )

def prepare_dataloader_tokenizer_training(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    """
    DataLoader without DistributedSampler, used if we are training
    a tokenizer in a single process or similar.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def prepare_datasets(args):
    """
    Prepare train/val datasets with optional RandAugment and normalization.
    """
    # rand_augment = RandAugment(num_ops=2, magnitude=9)

    postprocess_train = (
        transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),  # random crop + resize
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]),
        nn.Identity(),
    )

    postprocess_val = (
        transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]),
        nn.Identity(),
    )

    train_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=['jpg','cls'],
        train=True,
    ).map_tuple(*postprocess_train)

    val_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=['jpg','cls'],
        train=False,
    ).map_tuple(*postprocess_val)

    return train_dataset, val_dataset


def prepare_datasets_tokenizer_training(args):
    """
    A slightly different dataset pipeline if you are training a tokenizer.
    """
    postprocess_train = (
        transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]),
        nn.Identity(),
    )

    postprocess_val = (
        transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]),
        nn.Identity(),
    )

    train_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=['jpg','cls'],
        train=True,
    ).map_tuple(*postprocess_train)

    val_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=['jpg','cls'],
        train=False,
    ).map_tuple(*postprocess_val)

    return train_dataset, val_dataset


def freeze_vit_params(model, freeze_attn=True, freeze_ffn=True):
    """
    Example function to selectively freeze attention or MLP blocks
    in a Vision Transformer. Modify as appropriate for your model's attribute names.
    """
    for name, param in model.vit.named_parameters():
        if 'patch_embed' in name:
            continue
        if 'head' in name:
            continue
        if freeze_attn and 'attn' in name:
            param.requires_grad = False
        if freeze_ffn and 'mlp' in name:
            param.requires_grad = False
