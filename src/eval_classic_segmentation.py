import torch
import numpy as np
from skimage.segmentation import slic, watershed
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from metrics import explained_variance_batch
from train_utils import prepare_datasets
from input_parser import parse_input_args
import sys
from torch.utils.data import Dataset, DataLoader
import quixdata
import torchvision.transforms.v2 as transforms
import torch.nn as nn
from tqdm import tqdm  # <-- Added tqdm import

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
    )
    
def prepare_datasets(args):
    # Define the postprocessing transformations
    postprocess_train = (
        transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            #RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
        ]),
        nn.Identity(),
    )

    # Define the postprocessing transformations for validation
    postprocess_val = (
        transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]),
        nn.Identity(),
    )
    
    # Create the training dataset
    train_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=['jpg', 'cls'],
        train=True,
    ).map_tuple(*postprocess_train)

    # Create the validation dataset
    val_dataset = quixdata.QuixDataset(
        args.data_folder_name,
        args.data_subfolder_path,
        override_extensions=['jpg', 'cls'],
        train=False,
    ).map_tuple(*postprocess_val)
    
    return train_dataset, val_dataset

def run_segmentation_and_eval(train_loader,
                              val_loader,
                              device='cpu',
                              method='slic',
                              slic_n_segments=100,
                              slic_compactness=10,
                              watershed_threshold=0.5):
    """
    Runs segmentation (SLIC or watershed) on train/val data, then computes
    explained variance for each batch using the provided function.

    Args:
        train_loader, val_loader: PyTorch DataLoaders
        device: device to run on ('cpu' or 'cuda')
        method (str): 'slic' or 'watershed'
        slic_n_segments (int): Used if method='slic'
        slic_compactness (int): Used if method='slic'
        watershed_threshold (float): Example threshold if using watershed

    Returns:
        (train_ev_scores, val_ev_scores): lists of explained variance scores
        for each batch in train/val
    """

    train_ev_scores = []
    val_ev_scores = []

    def generate_labels(image_tensor):
        """
        image_tensor: PyTorch tensor of shape (C, H, W).
                      Values should be in range [0,1] or [0,255] as appropriate.
        returns: labels as a NumPy array of shape (H, W)
        """
        # Convert to NumPy for skimage: shape => (H, W, C)
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)

        if method.lower() == 'slic':
            # SLIC segmentation
            labels = slic(
                image_np,
                n_segments=slic_n_segments,
                compactness=slic_compactness,
                start_label=0
            )
            return labels

        elif method.lower() == 'watershed':
            # Convert to grayscale for watershed
            image_gray = rgb2gray(image_np)

            # Example marker creation: 2 regions (background & foreground)
            markers = np.zeros_like(image_gray, dtype=np.int32)
            markers[image_gray < watershed_threshold] = 1
            markers[image_gray > (1.0 - watershed_threshold)] = 2

            # Run watershed
            labels = watershed(image_gray, markers=markers)
            return labels

        else:
            raise ValueError(f"Unknown method: {method}")

    # --- Loop over TRAIN Loader ---
    print("Running superpixel segmentation on TRAIN set...")
    # Use tqdm for progress
    for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc="[TRAIN]")):
        # images: shape (batch_size, C, H, W)
        images = images.to(device)

        superpixel_labels_list = []
        for i in range(images.shape[0]):
            labels_np = generate_labels(images[i])
            labels_tensor = torch.from_numpy(labels_np).long()
            superpixel_labels_list.append(labels_tensor)

        # Stack into a single tensor of shape (batch_size, H, W)
        superpixel_labels_batch = torch.stack(superpixel_labels_list, dim=0).to(device)

        # Compute explained variance for this batch
        batch_ev = explained_variance_batch(images, superpixel_labels_batch)
        train_ev_scores.extend(batch_ev)

        # Print partial running results every 10th batch
        if (batch_idx + 1) % 10 == 0:
            partial_mean_ev = float(np.mean(train_ev_scores))
            print(f"[TRAIN] Batch {batch_idx+1}/{len(train_loader)} - "
                  f"Running Mean EV: {partial_mean_ev:.4f}")

    # --- Loop over VAL Loader ---
    print("Running superpixel segmentation on VALIDATION set...")
    for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="[VAL]")):
        images = images.to(device)

        superpixel_labels_list = []
        for i in range(images.shape[0]):
            labels_np = generate_labels(images[i])
            labels_tensor = torch.from_numpy(labels_np).long()
            superpixel_labels_list.append(labels_tensor)

        superpixel_labels_batch = torch.stack(superpixel_labels_list, dim=0).to(device)
        batch_ev = explained_variance_batch(images, superpixel_labels_batch)
        val_ev_scores.extend(batch_ev)

        # Print partial running results every 10th batch
        if (batch_idx + 1) % 10 == 0:
            partial_mean_ev = float(np.mean(val_ev_scores))
            print(f"[VAL]   Batch {batch_idx+1}/{len(val_loader)} - "
                  f"Running Mean EV: {partial_mean_ev:.4f}")

    # Compute final average explained variance
    train_mean_ev = float(np.mean(train_ev_scores)) if train_ev_scores else 0.0
    val_mean_ev = float(np.mean(val_ev_scores)) if val_ev_scores else 0.0

    print(f"[TRAIN] Final Mean Explained Variance: {train_mean_ev:.4f}")
    print(f"[VAL]   Final Mean Explained Variance: {val_mean_ev:.4f}")

    return train_ev_scores, val_ev_scores

def main(args):
    device = "cpu"

    args = parse_input_args(args)

    # Prepare datasets/dataloaders
    train_dataset, val_dataset = prepare_datasets(args)
    train_dataloader = prepare_dataloader(train_dataset, args.batch_size)
    val_dataloader = prepare_dataloader(val_dataset, args.batch_size)

    # 1) Using SLIC
    print("\n=== SLIC Segmentation ===")
    run_segmentation_and_eval(
        train_dataloader, 
        val_dataloader, 
        device=device, 
        method='slic',           
        slic_n_segments=196,
        slic_compactness=20
    )

    # 2) Using Watershed
    print("\n=== Watershed Segmentation ===")
    run_segmentation_and_eval(
        train_dataloader, 
        val_dataloader, 
        device=device, 
        method='watershed',
        watershed_threshold=0.4
    )
    
if __name__ == "__main__":
    main(sys.argv[1:])
