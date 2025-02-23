import torch


def explained_variance_batch(image_batch, superpixel_labels_batch):
    batch_size, num_channels, height, width = image_batch.shape
    explained_variance_scores = []

    for i in range(batch_size):
        image = image_batch[i]  # Shape: (C, H, W)
        superpixel_labels = superpixel_labels_batch[i]  # Shape: (H, W)

        # Ensure superpixel_labels is in shape (H, W)
        superpixel_labels = superpixel_labels.squeeze().to(image.device)

        # Flatten image and labels for computation
        image_flat = image.view(num_channels, height * width)
        labels_flat = superpixel_labels.view(height * width)

        # Compute total variance of the image across all channels
        total_variance = image_flat.var(dim=1, unbiased=False).mean().item()

        # Compute within-superpixel variance
        unique_labels = superpixel_labels.unique()
        num_superpixels = unique_labels.size(0)

        pixel_sums = torch.zeros((num_superpixels, num_channels), device=image.device)
        pixel_squares = torch.zeros((num_superpixels, num_channels), device=image.device)
        pixel_counts = torch.zeros(num_superpixels, device=image.device)

        for j, label in enumerate(unique_labels):
            mask = (labels_flat == label)
            if mask.sum() == 0:
                continue
            pixel_sums[j] = image_flat[:, mask].sum(dim=1)
            pixel_squares[j] = (image_flat[:, mask] ** 2).sum(dim=1)
            pixel_counts[j] = mask.sum()

        pixel_means = pixel_sums / pixel_counts.unsqueeze(1)
        pixel_variances = (pixel_squares / pixel_counts.unsqueeze(1)) - (pixel_means ** 2)
        within_variance = pixel_variances.mean().item()

        # Compute explained variance
        explained_variance = 1 - (within_variance / total_variance)
        explained_variance_scores.append(explained_variance)

    return explained_variance_scores
