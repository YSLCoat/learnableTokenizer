"""
python -m differentiableTokenizer.train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.optim import AdamW

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import random

from .datasets import BSDS500Dataset
from .metrics import explained_variance_batch

from model import DifferentiableSuperpixelTokenizer

from train_utils import (
    prepare_dataloader_tokenizer_training,
    prepare_datasets_tokenizer_training
)

parser = argparse.ArgumentParser(description="Train or evaluate superpixel model")

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
    "--batch_size", 
    default=30, 
    help="Batch size"
)
parser.add_argument(
    "--img_size", 
    default=224, 
    help="Image size"
)

args = parser.parse_args()

eval_on_bsds=False
if eval_on_bsds:
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset_train = BSDS500Dataset(root_dir=r'D:\Data\BSDS500\data', split='train', transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
else:
    train_dataset, val_dataset = prepare_datasets_tokenizer_training(args)
    
    train_dataloader = prepare_dataloader_tokenizer_training(train_dataset, args.batch_size, True, 0)
    val_dataloader = prepare_dataloader_tokenizer_training(val_dataset, args.batch_size, False, 0)
    
    
# Training setup
NUM_CLUSTERS = 196
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DifferentiableSuperpixelTokenizer(
    max_segments=NUM_CLUSTERS,
    n_channels=3,
    use_positional_embeddings=False,
    reconstruction=True,
    embed_dim=192,
    device=device,
    superpixel_algorithm='voronoi_propagation'
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
max_epochs = 10

# Training loop
for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    
    # Training phase
    model.train()
    running_loss = 0.0
    ev_accum = [] 

    # Wrap train_dataloader with tqdm for progress bar
    for batch_idx, source in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
        if not eval_on_bsds:
            source = source[0]
        source = source.to(device)
        optimizer.zero_grad()
        
        final_embeddings, reconstructed_img, segments = model(source)
        
        loss = loss_fn(reconstructed_img, source)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        ev_scores = explained_variance_batch(source, segments)
        batch_ev = np.mean(ev_scores)
        ev_accum.append(batch_ev)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Batch Explained Variance: {batch_ev:.4f}")
    
    avg_loss = running_loss / len(train_dataloader)
    avg_ev = np.mean(ev_accum)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch+1} Average Explained Variance: {avg_ev:.4f}")
    
    if not eval_on_bsds:
        # Evaluation phase
        model.eval()
        running_val_loss = 0.0
        val_ev_accum = []
        with torch.no_grad():
            # Wrap val_dataloader with tqdm for progress bar
            for batch_idx, val_source in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")):
                val_source = val_source[0]
                val_source = val_source.to(device)
                final_embeddings, reconstructed_img, segments = model(val_source)
                
                val_loss = loss_fn(reconstructed_img, val_source)
                running_val_loss += val_loss.item()
                
                val_ev_scores = explained_variance_batch(val_source, segments)
                batch_val_ev = np.mean(val_ev_scores)
                val_ev_accum.append(batch_val_ev)
                
        avg_val_loss = running_val_loss / len(val_dataloader)
        avg_val_ev = np.mean(val_ev_accum)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1} Validation Explained Variance: {avg_val_ev:.4f}")
        
    # Optional: visualize a sample of the reconstructed images after training epoch
    sample_idx = random.randint(0, source.size(0) - 1)
    source_img = source[sample_idx].detach().cpu()
    recon_img = reconstructed_img[sample_idx].detach().cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    source_img = source_img * std + mean
    recon_img = recon_img * std + mean

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(source_img.permute(1,2,0).clamp(0,1))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(recon_img.permute(1,2,0).clamp(0,1))
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()
    
# Save the model at the end of training
model_save_path = "differentiable_superpixel_tokenizer.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")