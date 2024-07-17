import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from tqdm import tqdm
import time
from datetime import timedelta
import quixdata


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

        
        self.results = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
    def _run_batch(self, source, targets, train=False):
        if train:
            self.model.train()  # Set the model to training mode
            self.optimizer.zero_grad()
        else:
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                output = self.model(source)
                loss = torch.nn.CrossEntropyLoss()(output, targets)
                return loss.item(), output.argmax(dim=1)

        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), output.argmax(dim=1)

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for source, targets in tqdm(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, preds = self._run_batch(source, targets, train=True)
            train_loss += loss
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        train_loss /= len(self.train_data)
        train_accuracy = correct_train / total_train
        
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        for source, targets in tqdm(self.val_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, preds = self._run_batch(source, targets, train=False)
            val_loss += loss
            correct_val += (preds == targets).sum().item()
            total_val += targets.size(0)
        
        val_loss /= len(self.val_data)
        val_accuracy = correct_val / total_val
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_accuracy:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_accuracy:.4f}"
        )
        
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
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
        sampler=DistributedSampler(dataset)
    )
    
def prepare_datasets(args):
    # Define the postprocessing transformations
    postprocess = (
        transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
    ).map_tuple(*postprocess)

    # Create the validation dataset
    val_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=[
            'jpg',
            'cls'
        ],
        train=False,
    ).map_tuple(*postprocess)
    
    return train_dataset, val_dataset
    