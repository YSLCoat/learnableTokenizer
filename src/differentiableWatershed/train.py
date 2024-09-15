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
from differentiableWatershed.model import VoronoiPropagation
from torchinfo import summary
from differentiableSlic.lib.dataset import augmentation
from utils import train, plot, calculate_warmup_epochs, DiceLoss
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_utils import BSDS
from loss_functions import combined_loss
import matplotlib.pyplot as plt
 
device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_predicted_segments(input_image, predictions, masks=None, num_classes=50):
    """
    Plot input image, predicted segments, and ground truth masks for the model.
    
    Args:
        input_image: torch.Tensor
            A Tensor of shape (batch_size, 3, height, width) for RGB input images.
        predictions: torch.Tensor
            A Tensor of shape (batch_size, 50, height, width). These are the raw model predictions.
        masks: torch.Tensor, optional
            Ground truth mask of shape (batch_size, 50, height, width) or (batch_size, height, width).
            If masks have multiple channels, they should be one-hot encoded, and we use argmax to convert them.
        num_classes: int
            The number of classes predicted by the model (default: 50).
    """
    # Get the predicted class for each pixel (argmax over the 50 classes)
    predicted_classes = torch.argmax(predictions, dim=1)  # Shape: (batch_size, height, width)
    
    # Converting to numpy for visualization
    predicted_classes_np = predicted_classes.cpu().detach().numpy()
    
    # Convert input image to numpy (assuming it's in range [0, 1] or [0, 255])
    input_image_np = input_image.cpu().detach().numpy()  # Shape: (batch_size, height, width, 3)
    
    # Plot the input image, predictions, and ground truth masks (if provided)
    batch_size = input_image_np.shape[0]
    for i in range(batch_size):
        plt.figure(figsize=(15, 5))
        
        # Plot input image
        plt.subplot(1, 3, 1)
        plt.imshow(input_image_np[i])  # Ensure the image is in (height, width, 3) format
        plt.title(f'Input Image (Sample {i+1})')
        plt.axis('off')
        
        # Plot predicted segments
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_classes_np[i], cmap='tab20', interpolation='none')
        plt.title(f'Predicted Segments (Sample {i+1})')
        plt.axis('off')
        
        # If ground truth mask is provided, plot the ground truth alongside
        if masks is not None:
            # Remove the extra batch dimension if necessary (squeeze)
            mask_sample = masks[i].squeeze(0)  # Remove batch dimension
            
            # Check if masks have multiple channels (e.g., (50, height, width))
            if mask_sample.shape[0] == num_classes:
                # Convert ground truth mask from multi-channel (50, height, width) to class index per pixel
                ground_truth_mask = torch.argmax(mask_sample, dim=0).cpu().detach().numpy()  # Shape: (height, width)
            else:
                # If ground truth mask is already single-channel (class indices), use it directly
                ground_truth_mask = mask_sample.cpu().detach().numpy()  # Shape: (height, width)

            plt.subplot(1, 3, 3)
            plt.imshow(ground_truth_mask, cmap='tab20', interpolation='none')
            plt.title(f'Ground Truth Mask (Sample {i+1})')
            plt.axis('off')
                
        plt.show()
                
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

    
    args = parser.parse_args()
    
    augment = augmentation.Compose([augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = BSDS(r'D:\Data', geo_transforms=augment)
    train_loader = DataLoader(train_dataset, 12, shuffle=True, drop_last=True, num_workers=0)

    test_dataset = BSDS(r'D:\Data', split="val")
    test_loader = DataLoader(test_dataset, 12, shuffle=False, drop_last=False)
    
    model = VoronoiPropagation(num_clusters=args.n_classes).to(device)
    #model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=args.n_classes).to(device)
    # scaler = torch.cuda.amp.GradScaler() # this will help us to use mixed precision training
    #summary(model, input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size), depth=4)

    optimizer = AdamW(model.parameters(), betas=[args.beta_1, args.beta_2], lr=args.lr, weight_decay=args.weight_decay)
    
    train_loss_list = []
    
    for epoch in range(args.epochs):
        ## Training phase 
        model.train()
        tk0 = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1} Training")

        train_loss = 0
        train_exapmles = 0

        # loop on the train loader
        for batch_idx, (images_raw, images, masks) in enumerate(tk0):
            images_raw = images_raw.to(device)
            images = images.to(device)
            masks = masks.to(device) # add a channel dimension to the mask (since it is a single channel image)
            height, width = images.shape[-2:]

            preds = model(images)
            loss = combined_loss(preds, masks)
            
            # print(loss)

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
            
            if epoch == 10:
                model.eval()
                with torch.no_grad():
                    predictions = model(images)

                # Visualize predictions and masks
                plot_predicted_segments(images_raw, predictions, masks=masks)

        ## Validation phase
        model.eval()
        tk1 = tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch+1} Validation")

        val_loss = 1
        val_examples = 1

        # with torch.no_grad():
        #     for batch_idx, (images, masks) in enumerate(tk1):
        #         images = images.to(device)
        #         masks = masks.to(device)

        #         preds = model(images)
            
        #         # calculate the loss
        #         #print(preds.shape, masks.shape)
        #         #print(torch.unique(masks))
                
        #         loss = combined_loss(preds, masks).float()
                
        #         val_loss += loss.item()
        #         val_examples += images.size(0)

        #         # update the progress bar
        #         tk1.set_postfix(loss=(val_loss/val_examples))

        # # save the model if the accuracy is improved
        # accuracy = check_accuracy_multiclass(test_loader, model, device, args.n_classes)
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "best_accuracy": best_accuracy,
        #     }
        #     torch.save(checkpoint, "model.pth")
        
        accuracy = 0

        print(f"Epoch {epoch+1}, train loss: {train_loss/train_exapmles}, val loss: {val_loss/val_examples}, val accuracy: {accuracy}")

        # save the predictions as images every 5 epochs
        if epoch % 5 == 0:
            pass#save_predictions_as_imgs(val_loader, model, device, folder="saved_images/")
        
        train_loss_list.append(train_loss/train_exapmles)
        
    plt.plot(train_loss_list)
    plt.show()
    
