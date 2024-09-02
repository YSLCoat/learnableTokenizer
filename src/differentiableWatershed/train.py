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
from utils import train, plot, calculate_warmup_epochs, DiceLoss
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_utils import BSDS500Dataset


import os
print(os.listdir("F:\data\Caravana"))

import zipfile
import shutil

DATASET_DIR = 'F:\data\Caravana\\'
WORKING_DIR = 'F:\data\Caravana\extracted\\'


def check_accuracy_multiclass(loader, model, device, num_classes):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            # Calculate Dice score for each class and average
            dice = 0
            for cls in range(num_classes):
                pred_cls = (preds == cls).float()
                true_cls = (y == cls).float()
                dice += (2 * (pred_cls * true_cls).sum()) / ((pred_cls + true_cls).sum() + 1e-8)
            dice_score += dice / num_classes

    print(
        f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}'
    )
    print(f'Dice score: {dice_score/len(loader)}')
    model.train()
    return dice_score / len(loader)


def save_predictions_as_imgs(loader, model, device, folder="saved_images/"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    num_examples = 0
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        for i in range(preds.size(0)):  # Iterate over each image in the batch
            torchvision.utils.save_image(preds[i], os.path.join(folder, f"pred_{idx}_{i}.png"))
            torchvision.utils.save_image(y[i].unsqueeze(0), os.path.join(folder, f"mask_{idx}_{i}.png"))  # unsqueeze adds a channel dimension to the tensor
            num_examples += 1
            if num_examples == 10:
                model.train()
                return
            
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data_subfolder_path", default=r"F:\data")
    
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
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_image_dir = r'F:\data\Caravana\extracted\train'
    train_mask_dir = r'F:\data\Caravana\extracted\train_masks'
    val_image_dir = r'F:\data\Caravana\extracted\val'
    val_mask_dir = r'F:\data\Caravana\extracted\val_masks'
    ## Dataset hyperparameters
    batch_size = 16
    image_height = 224
    image_width = 224
    pin_memory = True

    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            # this will only divide by 255 (since mean = 0 and std = 1)
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # for the validation transforms, we will only resize and normalize without any augmentations
    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # from data_utils import CarvanaDataset
    
    # train_dataset = CarvanaDataset(image_dir=train_image_dir,mask_dir=train_mask_dir, transform=train_transform)
    # val_dataset = CarvanaDataset(image_dir=val_image_dir,mask_dir=val_mask_dir, transform=val_transform)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])
    
    train_dataset = BSDS500Dataset(root_dir=r'D:\Data\BSDS500\data', split='train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    val_dataset = BSDS500Dataset(root_dir=r'D:\Data\BSDS500\data', split='val', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    model = DifferentiableWatershedWithVoronoi(num_markers=args.n_classes, num_iterations=6).to(device)
    # model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    scaler = torch.cuda.amp.GradScaler() # this will help us to use mixed precision training
    #summary(model, input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size), depth=4)

    optimizer = AdamW(model.parameters(), betas=[args.beta_1, args.beta_2], lr=args.lr, weight_decay=args.weight_decay)
    #loss_criterion = nn.CrossEntropyLoss() #DiceLoss()
    
    criterion = nn.CrossEntropyLoss()
    use_scheduler = False
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    # start = time.time()
    # #results = train(args, model, train_loader, val_loader, optimizer, loss_criterion, epochs=args.epochs, device=device, model_save_path=None)
    # end = time.time()

    # state_dict = {
    #     'model': model.state_dict(),
    #     'training_args': args,
    #     'training_time': end - start,
    # }
    
    for epoch in range(args.epochs):
        ## Training phase 
        model.train()
        tk0 = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1} Training")

        train_loss = 0
        train_exapmles = 0

        # loop on the train loader
        for batch_idx, (images, masks) in enumerate(tk0):
            images = images.to(device)
            masks = masks.float().to(device) # add a channel dimension to the mask (since it is a single channel image)

            
            #with torch.cuda.amp.autocast():
                # forward pass
            preds = model(images)
            # calculate the loss
            #print(preds.shape, masks.shape)
            #print(torch.unique(masks))
            loss = criterion(preds, masks.long())
            train_loss += loss.item()
            train_exapmles += images.size(0)
            
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward() # scale the loss to avoid underflow or overflow
            
            # update the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # update the progress bar
            tk0.set_postfix(loss=(train_loss/train_exapmles))

        ## Validation phase
        model.eval()
        tk1 = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1} Validation")

        val_loss = 0
        val_examples = 0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tk1):
                images = images.to(device)
                masks = masks.float().to(device)

                # forward pass
                preds = model(images)

                # calculate the loss
                loss = criterion(preds, masks.long())
                val_loss += loss.item()
                val_examples += images.size(0)

                # update the progress bar
                tk1.set_postfix(loss=(val_loss/val_examples))

        # save the model if the accuracy is improved
        accuracy = check_accuracy_multiclass(val_loader, model, device, args.n_classes)
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "best_accuracy": best_accuracy,
        #     }
        #     torch.save(checkpoint, "model.pth")

        print(f"Epoch {epoch+1}, train loss: {train_loss/train_exapmles}, val loss: {val_loss/val_examples}, val accuracy: {accuracy}")

        # save the predictions as images every 5 epochs
        if epoch % 5 == 0:
            pass#save_predictions_as_imgs(val_loader, model, device, folder="saved_images/")
        
        if use_scheduler:
            scheduler.step() 

