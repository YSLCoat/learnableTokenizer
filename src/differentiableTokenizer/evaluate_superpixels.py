import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

from PIL import Image  # for loading a single image

from skimage.segmentation import slic, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.segmentation import mark_boundaries

from datasets import BSDS500Dataset
from metrics import explained_variance_batch
from model import DifferentiableSuperpixelTokenizer
# from train_utils import prepare_dataloader_tokenizer_training, prepare_datasets_tokenizer_training

import torch.nn as nn

def prepare_dataloader_tokenizer_training(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        num_workers=num_workers,
    )


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
    

def reconstruct_from_segments(image, segments):
    """
    Given an image (H, W, C) and segmentation labels (H, W), compute a 
    reconstructed image by replacing each segment with its mean color.
    """
    reconstructed = np.zeros_like(image)
    for seg_val in np.unique(segments):
        mask = segments == seg_val
        mean_color = image[mask].mean(axis=0)
        reconstructed[mask] = mean_color
    return reconstructed

def reconstruct_from_segments_torch(image, segments):
    """
    Given an image (H, W, C) that is normalized by ImageNet stats
    (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    compute a reconstructed image by replacing each segment
    with its mean color in *unnormalized* space.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, C)
        The input image, normalized by ImageNet coefficients.
    segments : np.ndarray, shape (H, W)
        An integer label mask defining superpixel segments.

    Returns
    -------
    reconstructed : np.ndarray, shape (H, W, C)
        The unnormalized reconstruction, clipped to [0, 1].
    """

    # 1) Unnormalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    unnorm_image = image * std + mean  # broadcast each channel: image[..., c] = image[..., c]*std[c] + mean[c]

    # 2) Reconstruct by segment-wise mean color
    reconstructed = np.zeros_like(unnorm_image)
    unique_segments = np.unique(segments)
    for seg_val in unique_segments:
        mask = (segments == seg_val)
        mean_color = unnorm_image[mask].mean(axis=0)  # shape (3,)
        reconstructed[mask] = mean_color

    # 3) Clip to ensure [0,1]
    reconstructed = np.clip(reconstructed, 0, 1)
    return reconstructed

def compute_explained_variance(image, reconstructed):
    """
    Compute the explained variance score:
        explained_variance = 1 - (variance(residual) / variance(original))
    Flatten the image to take all channels into account.
    """
    image_flat = image.reshape(-1)
    recon_flat = reconstructed.reshape(-1)
    total_var = np.var(image_flat)
    residual_var = np.var(image_flat - recon_flat)
    return 1 - residual_var / total_var if total_var > 0 else 1.0

def evaluate_pytorch_model(model, dataloader, device):
    """
    Evaluate the PyTorch model using the provided dataloader and
    the pre-defined explained_variance_batch function.
    """
    model.eval()
    ev_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PyTorch model"):
            # Assume the dataloader returns a tuple (images, ...)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            # Forward pass: obtain segmentation and reconstructed image
            final_embeddings, reconstructed_img, segments, gradient_map = model(images)
            # Compute explained variance using your provided metric
            ev_scores = explained_variance_batch(images, segments)
            ev_list.append(np.mean(ev_scores))
    return np.mean(ev_list)

def evaluate_scikit_method(method, dataloader):
    """
    Evaluate a scikit-image based segmentation method (SLIC or Watershed).
    For each image, the method computes a segmentation, then reconstructs
    the image from segment-wise mean colors, and finally calculates the 
    explained variance.
    """
    ev_list = []
    for batch in tqdm(dataloader, desc=f"Evaluating {method}"):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        # images: tensor of shape (B, C, H, W) assumed to be in [0,1]
        images_np = images.permute(0, 2, 3, 1).numpy()
        for img in images_np:
            if method == "slic":
                # Use SLIC to generate ~196 superpixels (adjust parameters as needed)
                segments = slic(img, n_segments=196, compactness=10, start_label=0)
            elif method == "watershed":
                # Compute gradient using sobel on a grayscale version
                gradient = sobel(rgb2gray(img))
                # Identify local minima as markers
                coordinates = peak_local_max(-gradient, min_distance=5)
                markers = np.zeros(gradient.shape, dtype=int)
                for i, (r, c) in enumerate(coordinates, 1):
                    markers[r, c] = i
                segments = watershed(gradient, markers, mask=np.ones(gradient.shape, dtype=bool))
            else:
                raise ValueError(f"Unknown method: {method}")
            reconstructed = reconstruct_from_segments(img, segments)
            ev = compute_explained_variance(img, reconstructed)
            ev_list.append(ev)
    return np.mean(ev_list)


def visualize_segments(
    image_path: str,
    method: str,
    model_path: str = None,
    device: str = "cpu",
    img_size: int = 224,
    num_segments: int = 196
):
    """
    Load a single image from `image_path`, segment it using the specified `method`,
    and visualize the following:
      1) The original image (resized)
      2) The superpixel boundaries (via mark_boundaries)
      3) The reconstructed image
      4) The gradient map (if available)

    Parameters:
    -----------
    image_path : str
        Path to the input image.
    method : str
        One of ['slic', 'watershed', 'pytorch'].
    model_path : str, optional
        Path to the PyTorch model checkpoint (required if `method=='pytorch'`).
    device : str, optional
        Device to run the model on ('cpu' or 'cuda').
    img_size : int, optional
        Size at which to resize the image for segmentation.
    num_segments : int, optional
        Number of segments to use for SLIC or the PyTorch model.
    """

    # 1) Load image with PIL and convert to RGB
    pil_img = Image.open(image_path).convert('RGB')

    # 2) Prepare transforms
    if method == 'pytorch':
        # Same transform as training (with normalization)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    else:
        # SLIC or Watershed: keep values in [0,1], no normalization
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    # Apply transform
    input_tensor = transform(pil_img)  # (C, H, W), values in [0,1] (if scikit)
    input_batch = input_tensor.unsqueeze(0)  # (1, C, H, W)

    # Convert to numpy for scikit or reconstruction operations
    # shape (H, W, C)
    img_np = input_tensor.permute(1, 2, 0).numpy()

    # 3) Segment the image & produce the gradient map
    segments = None
    gradient_map = None
    reconstructed = None

    if method == 'pytorch':
        if model_path is None:
            raise ValueError("Please provide model_path when method='pytorch'.")

        # Load the model
        model = DifferentiableSuperpixelTokenizer(
            max_segments=num_segments,
            n_channels=3,
            use_positional_embeddings=False,
            reconstruction=True,
            embed_dim=192,
            device=device,
            superpixel_algorithm="slic_segmentation"
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            input_batch = input_batch.to(device)
            # The model returns:
            #  final_embeddings, reconstructed_img, segments, gradient_map
            _, _, segs, grad_map = model(input_batch)
            segments = segs[0].cpu().numpy()  # shape (H, W)
            # The gradient_map might be (B,1,H,W) or (B,H,W). Adjust as needed
            grad_map = grad_map[0].cpu().numpy()
            if len(grad_map.shape) == 3:
                grad_map = grad_map[0]  # remove channel dim if present
            gradient_map = grad_map  # shape (H, W)

        # Reconstruct via our existing function
        reconstructed = reconstruct_from_segments_torch(img_np, segments)

    elif method == 'watershed':
        # We do the sobel gradient in grayscale
        gray = rgb2gray(img_np)
        grad_map = sobel(gray)
        # Identify local minima as markers
        coordinates = peak_local_max(-grad_map, min_distance=5)
        markers = np.zeros(grad_map.shape, dtype=int)
        for i, (r, c) in enumerate(coordinates, 1):
            markers[r, c] = i
        segments = watershed(grad_map, markers, mask=np.ones(grad_map.shape, dtype=bool))

        gradient_map = grad_map
        reconstructed = reconstruct_from_segments(img_np, segments)

    elif method == 'slic':
        segments = slic(img_np, n_segments=num_segments, compactness=10, start_label=0)
        # For gradient map, let's do a simple Sobel on the grayscale
        gradient_map = sobel(rgb2gray(img_np))
        reconstructed = reconstruct_from_segments(img_np, segments)

    else:
        raise ValueError("Unknown method. Choose from ['slic', 'watershed', 'pytorch'].")

    # 4) Create a "marked boundaries" image
    #    We overlay red boundaries on top of the (resized) image
    #    `mark_boundaries` expects the image in [0,1] or [0,255], but scikit usually
    #    expects a floating image in [0,1], so for PyTorch (normalized) we should clamp + shift
    #    if needed. But since we're already in [0,1] for scikit methods, it's fine.
    #    For the PyTorch method, we used normalization for the model. 
    #    However, `img_np` is AFTER reversing that normalization or not?
    #
    #    If you used ImageNet normalization for PyTorch, your img_np won't be in [0,1].
    #    If you want the "visual" look in correct color, you can re-normalize. 
    #    For simplicity, let's assume it's okay to just clamp it for display:
    #
    if method == 'pytorch':
        # De-normalize to a typical [0,1] range for display
        # (C, H, W) => (H, W, C)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        # Reverse normalization
        img_np_for_mark = (img_np * std[None, None, :]) + mean[None, None, :]
        # clamp to [0,1]
        img_np_for_mark = np.clip(img_np_for_mark, 0, 1)
    else:
        # SLIC / watershed => already [0,1]
        img_np_for_mark = img_np

    marked_image = mark_boundaries(
        img_np_for_mark, 
        segments, 
        color=(1, 0, 0),  # Red boundaries
        mode='outer'
    )

    # 5) Plot all in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Top-Left: Resized image
    axs[0, 0].imshow(np.clip(img_np_for_mark, 0, 1))
    axs[0, 0].set_title("Resized Image")
    axs[0, 0].axis("off")

    # Top-Right: Marked boundaries
    axs[0, 1].imshow(marked_image)
    axs[0, 1].set_title("Marked Boundaries")
    axs[0, 1].axis("off")

    # Bottom-Left: Reconstructed
    axs[1, 0].imshow(reconstructed)
    axs[1, 0].set_title("Reconstructed")
    axs[1, 0].axis("off")

    # Bottom-Right: Gradient Map
    # If we don't have a gradient map, just disable the axis
    if gradient_map is not None:
        axs[1, 1].imshow(gradient_map, cmap='gray')
        axs[1, 1].set_title("Gradient Map")
        axs[1, 1].axis("off")
    else:
        axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare superpixel segmentation methods"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["pytorch", "slic", "watershed"],
        help="Segmentation method to evaluate: 'pytorch', 'slic', or 'watershed'"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the PyTorch model checkpoint (required for method 'pytorch')"
    )
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for resizing evaluation images")
    parser.add_argument("--data_path", type=str, default=r"F:\data", help="Path to the data folder/subfolder")
    parser.add_argument("--data_folder_name", type=str, default=r"IN1k", help="Name of the data folder")
    parser.add_argument(
        "--use_bsds",
        action="store_true",
        help="If set, use the BSDS500 dataset for evaluation."
    )
    parser.add_argument(
        "--visualize_image_path",
        type=str,
        default=None,
        help="If provided, will visualize the segmentation results for this single image."
    )
    args = parser.parse_args()

    if args.method == "pytorch" and args.model_path is None:
        parser.error("--model_path is required when method is 'pytorch'")

    device = 'cpu'#"cuda" if torch.cuda.is_available() else "cpu"

    if args.visualize_image_path:
        # For demonstration (and to ensure it doesn't break the standard evaluation)
        visualize_segments(
            image_path=args.visualize_image_path,
            method=args.method,
            model_path=args.model_path,
            device=device,
            img_size=args.img_size,
            num_segments=196  # or any default number of segments you prefer
        )
        return  # Exit after visualization

    if args.use_bsds:
        if args.method == "pytorch":
            transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor()  # values in [0,1]
            ])
        dataset = BSDS500Dataset(root_dir=args.data_path, split="test", transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        train_dataset, val_dataset = prepare_datasets_tokenizer_training(args)
        if args.method == "pytorch":
            dataloader = prepare_dataloader_tokenizer_training(val_dataset, args.batch_size, False, 0)
        else:
            val_dataset.transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
            ])
            dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.method == "pytorch":
        NUM_CLUSTERS = 196 
        model = DifferentiableSuperpixelTokenizer(
            max_segments=NUM_CLUSTERS,
            n_channels=3,
            use_positional_embeddings=False,
            reconstruction=True,
            embed_dim=192,
            device=device,
            superpixel_algorithm="voronoi_propagation"
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded PyTorch model from {args.model_path}")
        ev_score = evaluate_pytorch_model(model, dataloader, device)
    else:
        ev_score = evaluate_scikit_method(args.method, dataloader)

    print(f"Average explained variance for method '{args.method}': {ev_score:.4f}")

if __name__ == "__main__":
    main()
