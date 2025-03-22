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
from utils import load_model_from_state_dict, visualize_segmentation_and_reconstruction
from metrics import explained_variance_batch


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
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
        # scheduler: torch.optim.lr_scheduler._LRScheduler,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.save_every = save_every
        self.mixup_augmentation = mixup_augmentation(args.n_classes)
        self.verbose_training = True
        
        if args.train_from_checkpoint:
            self.model = load_model_from_state_dict(self.model, args.pretrained_model_path)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=False)
        
        self.results = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        self.args = args
        
    def _run_batch(self, source, targets, train=False):
        if train:
            self.model.train()  # Set the model to training mode
            self.optimizer.zero_grad()
            
            final_embeddings, reconstructed_img, segments, gradient_map = self.model(source)
            loss = self.loss_function(reconstructed_img, source)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # self.scheduler.step()
            
            return loss.item(), reconstructed_img, segments, gradient_map
        else:
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                final_embeddings, reconstructed_img, segments, gradient_map = self.model(source)
                loss = self.loss_function(reconstructed_img, source)
                return loss.item(), reconstructed_img, segments, gradient_map

    def _run_epoch(self, epoch):
        # Get batch size from the first batch of train_data
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        
        train_loss = 0.0
        train_ev_total = 0.0
        num_train_batches = 0
        for batch_idx, (source, targets) in enumerate(tqdm(self.train_data)):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, preds, segments, gradient_map = self._run_batch(source, targets, train=True)
            train_loss += loss

            # Compute explained variance for the batch
            # Here, 'source' is treated as the image batch and 'segments' as the corresponding superpixel labels.
            mean = torch.tensor([0.485, 0.456, 0.406], device=source.device).view(1, -1, 1, 1)  # (1, C, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=source.device).view(1, -1, 1, 1)
            unnormalized_batch = (source * std) + mean
            unnormalized_reconstruct = (preds * std) + mean
            batch_ev_scores = explained_variance_batch(unnormalized_batch, segments)
            batch_ev = sum(batch_ev_scores) / len(batch_ev_scores)
            train_ev_total += batch_ev
            num_train_batches += 1

            if self.verbose_training:
                print(f"Epoch: {epoch+1} | Batch: {batch_idx+1} | Loss: {loss:.4f} | EV: {batch_ev:.4f}")
                
            if batch_idx == 1 or batch_idx % 10 == 0:  # Adjust the frequency as needed
                # Choose a sample from the batch (for example, the first sample)
                sample_image = unnormalized_batch[0]          # shape: [C, H, W]
                sample_gradient = gradient_map[0]   # shape: [H, W] (or [1, H, W])
                sample_segments = segments[0]       # shape: [H, W]
                sample_reconstructed = unnormalized_reconstruct[0]
                
                # Define the output folder and filename.
                output_folder = "visualizations_" + str(self.args.superpixel_algorithm)
                filename = f"epoch_{epoch+1}_batch_{batch_idx+1}.png"
                visualize_segmentation_and_reconstruction(sample_image, sample_gradient, sample_segments, sample_reconstructed, output_folder, filename)

        train_loss /= len(self.train_data)
        train_ev = train_ev_total / num_train_batches

        val_loss = 0.0
        val_ev_total = 0.0
        num_val_batches = 0
        for batch_idx, (source, targets) in enumerate(tqdm(self.val_data)):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, preds, segments, gradient_map = self._run_batch(source, targets, train=False)
            val_loss += loss

            mean = torch.tensor([0.485, 0.456, 0.406], device=source.device).view(1, -1, 1, 1)  # (1, C, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=source.device).view(1, -1, 1, 1)
            unnormalized_batch = (source * std) + mean
            batch_ev_scores = explained_variance_batch(unnormalized_batch, segments)
            batch_ev = sum(batch_ev_scores) / len(batch_ev_scores)
            val_ev_total += batch_ev
            num_val_batches += 1

            if self.verbose_training:
                print(f"Val Epoch: {epoch+1} | Batch: {batch_idx+1} | Loss: {loss:.4f} | EV: {batch_ev:.4f}")

        val_loss /= len(self.val_data)
        val_ev = val_ev_total / num_val_batches

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_ev: {train_ev:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_ev: {val_ev:.4f}"
        )

        
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "learnableGradMapTokenizer_" + str(self.args.superpixel_algorithm) + "_.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            start = time.time()
            self._run_epoch(epoch)
            end = time.time()
            if epoch==0:
                elapsed_time_seconds = end - start
                elapsed_time_formatted = str(timedelta(seconds=int(elapsed_time_seconds)))
                print(f"First epoch took: {elapsed_time_formatted}. Estimated training time: {str(timedelta(seconds=int(elapsed_time_seconds * max_epochs)))}.")
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
                
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=4,
    )
    
def prepare_dataloader_tokenizer_training(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    
def prepare_datasets(args):
    # Define the postprocessing transformations
    postprocess_train = (
        transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            #RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]),
        nn.Identity(),
    )

    # Define the postprocessing transformations for validation
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
    
    # Create the training dataset
    train_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=[
            'jpg',
            'cls'
        ],
        train=True,
    ).map_tuple(*postprocess_train)

    # Create the validation dataset
    val_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=[
            'jpg',
            'cls'
        ],
        train=False,
    ).map_tuple(*postprocess_val)
    
    return train_dataset, val_dataset


def prepare_datasets_tokenizer_training(args):
    # Define the postprocessing transformations
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

    # Define the postprocessing transformations for validation
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
    
    # Create the training dataset
    train_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=[
            'jpg',
            'cls'
        ],
        train=True,
    ).map_tuple(*postprocess_train)

    # Create the validation dataset
    val_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=[
            'jpg',
            'cls'
        ],
        train=False,
    ).map_tuple(*postprocess_val)
    
    return train_dataset, val_dataset
    