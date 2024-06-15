import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import time
from datetime import timedelta
import math

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for X, y in tqdm(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """vals a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a valing dataset.

    Args:
        model: A PyTorch model to be valed.
        dataloader: A DataLoader instance for the model to be valed on.
        loss_fn: A PyTorch loss function to calculate loss on the val data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of valing loss and valing accuracy metrics.
        In the form (val_loss, val_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup val loss and val accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for X, y in tqdm(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

def train(args,
          model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          model_save_path) -> Dict[str, List]:
    """Trains and vals a PyTorch model.

    Passes a target PyTorch models through train_step() and val_step()
    functions for a number of epochs, training and validatinging the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and validateded.
        train_dataloader: A DataLoader instance for the model to be trained on.
        val_dataloader: A DataLoader instance for the model to be validated on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and validate loss as well as training and
        validate accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    val_loss: [...],
                    val_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    val_loss: [1.2641, 1.5706],
                    val_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_loss = float('inf')

    # Loop through training and valing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        #scheduler.step()
        start = time.time()
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        val_loss, val_acc = val_step(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with val loss: {best_val_loss:.4f}.")
            
        end = time.time()
        if epoch==0:
            elapsed_time_seconds = end - start
            elapsed_time_formatted = str(timedelta(seconds=int(elapsed_time_seconds)))
            print(f"First epoch took: {elapsed_time_formatted}. Estimated training time: {str(timedelta(seconds=int(elapsed_time_seconds * epochs)))}.")

    # Return the filled results at the end of the epochs
    return results

def plot(results, output_path):
    fig_loss = go.Figure()
    fig_acc = go.Figure()

    train_loss_trace = go.Scatter(x=list(range(len(results["train_loss"]))), y=results["train_loss"], mode='lines', name='Train Loss')
    val_loss_trace = go.Scatter(x=list(range(len(results["val_loss"]))), y=results["val_loss"], mode='lines', name='Validation Loss')

    train_acc_trace = go.Scatter(x=list(range(len(results["train_acc"]))), y=results["train_acc"], mode='lines', name='Train Accuracy')
    val_acc_trace = go.Scatter(x=list(range(len(results["val_acc"]))), y=results["val_acc"], mode='lines', name='Validation Accuracy')

    fig_loss.add_trace(train_loss_trace)
    fig_loss.add_trace(val_loss_trace)
    fig_loss.update_layout(title='Training and Validation Loss',
                           xaxis=dict(title='Epoch'),
                           yaxis=dict(title='Loss'))

    fig_acc.add_trace(train_acc_trace)
    fig_acc.add_trace(val_acc_trace)
    fig_acc.update_layout(title='Training and Validation Accuracy',
                          xaxis=dict(title='Epoch'),
                          yaxis=dict(title='Accuracy'))

    fig_loss.write_image(output_path + "loss_plot.png")
    fig_acc.write_image(output_path + "accuracy_plot.png")
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def calculate_warmup_epochs(dataset_size, batch_size, total_forward_passes):
    # Calculate the number of batches per epoch
    batches_per_epoch = math.ceil(dataset_size / batch_size)

    # Calculate the total number of epochs needed
    total_epochs = total_forward_passes / batches_per_epoch

    return total_epochs

def verify_model_name(model_name):
    # List of ViT models with image size 224
    vit_models_224 = [
        'vit_tiny_patch16_224',
        'vit_small_patch16_224',
        'vit_base_patch16_224',
        'vit_large_patch16_224',
        'vit_huge_patch14_224'
    ]
    
    # Check if the provided model name is in the list
    if model_name in vit_models_224:
        print("Loading model config for: ", model_name)
    else:
        assert 0, "Model configuration not found in timm."