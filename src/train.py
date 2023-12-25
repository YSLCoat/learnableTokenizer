"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
import torch.nn as nn
import time
import torchvision
from torchvision import transforms
from torch.optim import AdamW
from vision_transformer import VisionTransformer
from torchinfo import summary
import litdata.litdata as litdata

from utils import train, plot

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data_subfolder_path", default=r"F:\data")
    parser.add_argument("--data_folder_name", default=r"IN1k")
    parser.add_argument("--lr_scheduler", default=True, type=bool)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    
    parser.add_argument("--model_name", default="ViT_Base", type=str)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--num_attention_heads", default=12, type=int)
    parser.add_argument("--num_transformer_blocks", default=12, type=int)
    
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--n_channels", default=3, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    
    parser.add_argument("--train_cifar_100", action='store_true')
    

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    args = parser.parse_args()


    postprocess = (
    torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    nn.Identity(),
    )
    if args.train_cifar_100:
        
        transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),  # Resize and crop the image to 224x224 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])
                # Download and load the CIFAR-100 training dataset
        train_dataset = torchvision.datasets.CIFAR10(root='F:\data', train=True, download=True, transform=transform)

        # Download and load the CIFAR-100 testing dataset
        val_dataset = torchvision.datasets.CIFAR10(root='F:\data', train=False, download=True, transform=transform)
    else:            
        train_dataset = litdata.LITDataset(
            args.data_folder_name,
            args.data_subfolder_path,
            override_extensions=[
                'jpg',
                'cls'
            ],
            train = True,
        ).map_tuple(*postprocess)

        val_dataset = litdata.LITDataset(
            args.data_folder_name,
            args.data_subfolder_path,
            override_extensions=[
                'jpg',
                'cls'
            ],
            train = False,
        ).map_tuple(*postprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = VisionTransformer(args.patch_size, args.embed_dim, args.num_attention_heads, args.num_transformer_blocks, channels=args.n_channels, num_classes=args.n_classes).to(device)
    summary(model, input_size=(args.batch_size, args.n_channels, args.img_size, args.img_size), depth=4)

    output_file_path_best_model = os.path.join(output_folder, args.model_name + "_bestModel" + ".pt")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_criterion = nn.CrossEntropyLoss()
    start = time.time()
    results = train(args, model, train_loader, val_loader, optimizer, loss_criterion, epochs=args.epochs, device=device, model_save_path=output_file_path_best_model)
    end = time.time()

    state_dict = {
        'model': model.state_dict(),
        'training_args': args,
        'training_time': end - start,
    }
    output_file_path = os.path.join(output_folder, args.model_name + "_epochs_" + str(args.epochs) + ".pt")

    torch.save(state_dict, output_file_path)
    print(f"Model saved as {args.model_name}")

    plot(results, output_folder, args)