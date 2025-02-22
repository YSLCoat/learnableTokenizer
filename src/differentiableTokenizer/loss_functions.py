import torch
import torch.nn.functional as F

def softmax_predictions(predictions):
    """
    Apply softmax across the 50 predicted segments.
    Args:
        predictions: torch.Tensor
            A Tensor of shape (batch_size, 50, height, width)
    Returns:
        Softmaxed predictions with the same shape.
    """
    return F.softmax(predictions, dim=1)


def boundary_loss(predictions, masks):
    """
    Compute boundary-aware loss to encourage sharper edges.
    Args:
        predictions: torch.Tensor
            A Tensor of shape (batch_size, num_classes, height, width)
        masks: torch.Tensor
            Ground truth masks of shape (batch_size, num_classes, height, width)
    Returns:
        Boundary-aware loss.
    """
    
    # Remove any unnecessary singleton dimensions from masks
    masks = masks.squeeze()  # Remove any dimensions of size 1
    
    # If the masks have 50 channels (multi-class), reduce to a single channel
    if masks.dim() == 4 and masks.shape[1] == 50:  # Assuming second dimension is for classes
        masks = torch.argmax(masks, dim=1, keepdim=True)  # Convert multi-class mask to single channel
    
    # Convert predictions to a single-channel segmentation map using argmax
    pred_argmax = torch.argmax(predictions, dim=1, keepdim=True)  # Shape: (batch_size, 1, height, width)
    
    # Sobel filter to detect edges
    sobel_filter = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(predictions.device)

    # Apply Sobel filter to the single-channel predictions and masks
    pred_edges = F.conv2d(pred_argmax.float(), sobel_filter, padding=1)
    
    # Ensure masks are now (batch_size, 1, height, width)
    if masks.dim() == 3:  # If it lacks a channel dimension
        masks = masks.unsqueeze(1)  # Add the channel dimension

    # Apply Sobel filter to the masks
    mask_edges = F.conv2d(masks.float(), sobel_filter, padding=1)
    
    # Compute the L1 loss between the edges of predictions and masks
    return F.l1_loss(pred_edges, mask_edges)


def cross_entropy_reconstruction_loss(predictions, masks):
    """
    Cross-entropy reconstruction loss for segmentation.
    Args:
        predictions: torch.Tensor
            A Tensor of shape (batch_size, 50, height, width)
        masks: torch.Tensor
            A Tensor of shape (batch_size, 1, 50, height, width)
    Returns:
        Cross-entropy reconstruction loss.
    """
    predictions = softmax_predictions(predictions)
    
    masks = masks.squeeze(1)  # Convert (batch_size, 1, 50, height, width) -> (batch_size, 50, height, width)
    
    # Compute cross-entropy loss pixel-wise
    # assume `masks` is a one-hot encoded representation of the true segmentation labels
    loss = -(masks * predictions.log()).sum(1).mean()  # Summing across the 50 segments and taking the mean
    return loss

def mse_reconstruction_loss(predictions, masks):
    """
    MSE reconstruction loss for segmentation.
    Args:
        predictions: torch.Tensor
            A Tensor of shape (batch_size, 50, height, width)
        masks: torch.Tensor
            A Tensor of shape (batch_size, 1, 50, height, width)
    Returns:
        Mean squared error loss.
    """
    predictions = softmax_predictions(predictions)
    
    masks = masks.squeeze(1)  # Convert (batch_size, 1, 50, height, width) -> (batch_size, 50, height, width)
    
    # Compute MSE loss between the predicted segments and the ground truth masks
    loss = F.mse_loss(predictions, masks)
    return loss

def combined_loss(predictions, masks, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Combined loss function that combines cross-entropy and MSE reconstruction losses.
    Args:
        predictions: torch.Tensor
            A Tensor of shape (batch_size, 50, height, width)
        masks: torch.Tensor
            A Tensor of shape (batch_size, 1, 50, height, width)
        alpha: float
            Weight for the cross-entropy loss.
        beta: float
            Weight for the MSE loss.
    Returns:
        Combined loss.
    """
    ce_loss = cross_entropy_reconstruction_loss(predictions, masks)
    mse_loss = mse_reconstruction_loss(predictions, masks)
    boundary_loss_value = boundary_loss(predictions, masks)

    return alpha * ce_loss + beta * mse_loss + gamma * boundary_loss_value