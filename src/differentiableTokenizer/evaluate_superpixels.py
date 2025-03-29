import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

from skimage.segmentation import slic, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.feature import peak_local_max

from .datasets import BSDS500Dataset
from .metrics import explained_variance_batch
from model import DifferentiableSuperpixelTokenizer
from train_utils import prepare_dataloader_tokenizer_training, prepare_datasets_tokenizer_training


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
            _, _, segments = model(images)
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
                coordinates = peak_local_max(-gradient, min_distance=5, indices=True)
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
    args = parser.parse_args()

    if args.method == "pytorch" and args.model_path is None:
        parser.error("--model_path is required when method is 'pytorch'")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        dataset = BSDS500Dataset(root_dir=args.data_path, split="val", transform=transform)
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
