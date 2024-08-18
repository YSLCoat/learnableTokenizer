"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
import torch.nn as nn
import time
import torchvision
import sys
sys.path.append(r'D:\MSc_Data_Science_UiO\Thesis\learnableTokenizer\src')
from torchvision import transforms
from torch.optim import AdamW
from differentiableWatershed.model import DifferentiableWatershedWithVoronoi
from torchinfo import summary
from differentiableSlic.lib.dataset.bsds import BSDS
from differentiableSlic.lib.dataset import augmentation
from utils import train, plot, calculate_warmup_epochs
import segmentation_models_pytorch as smp


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data_subfolder_path", default=r"F:\data")
    parser.add_argument("--data_folder_name", default=r"IN1k")
    
    parser.add_argument("--lr_scheduler", default=True, type=bool)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--n_channels", default=3, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    
    args = parser.parse_args()

    augment = augmentation.Compose([augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = BSDS(args.data_subfolder_path, geo_transforms=augment)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    test_dataset = BSDS(args.data_subfolder_path, split="val", geo_transforms=augment)
    val_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False, drop_last=False)
    
    model = DifferentiableWatershedWithVoronoi(num_markers=50, num_iterations=5).to(device)
    # model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=50).to(device)

    #summary(model, input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size), depth=4)

    optimizer = AdamW(model.parameters(), betas=[args.beta_1, args.beta_2], lr=args.lr, weight_decay=args.weight_decay)
    loss_criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    results = train(args, model, train_loader, val_loader, optimizer, loss_criterion, epochs=args.epochs, device=device, model_save_path=None)
    end = time.time()

    state_dict = {
        'model': model.state_dict(),
        'training_args': args,
        'training_time': end - start,
    }
    output_file_path = os.path.join(output_folder, args.model_name + "_epochs_" + str(args.epochs) + ".pt")

    torch.save(state_dict, output_file_path)
    print(f"Model saved as {args.model_name}")

    plot(results, output_file_path)