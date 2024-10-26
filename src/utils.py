import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import time
from datetime import timedelta
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn as nn
import os

mean = torch.tensor([0.485, 0.456, 0.406]).to("cuda")
std = torch.tensor([0.229, 0.224, 0.225]).to("cuda")

def get_available_gpus():
    if torch.cuda.is_available():
        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        # Print the name of each available GPU
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")

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
    total_pixels = 0
    correct_pixels = 0

    # Loop through data loader data batches
    for X, y in tqdm(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        X = (X - mean[None, :, None, None]) / std[None, :, None, None]
        # 1. Forward pass
        markers = model(X)

        y_pred = markers.reshape(y.shape[0], y.shape[1], X.shape[2]*X.shape[3]).float()

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Calculate accuracy
        # Assuming y and y_pred are both (batch_size, channels, height * width)
        # y_pred_bin = torch.round(y_pred)  # Threshold the predictions to get binary values (0 or 1)
        # correct_pixels += (y_pred_bin == y).sum().item()
        # total_pixels += y.numel()
        
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = 0 # correct_pixels / total_pixels 
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
            X = (X - mean[None, :, None, None]) / std[None, :, None, None]

            # 1. Forward pass
            val_pred_logits = model(X).reshape(y.shape[0], 50, 50176).float()

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            # val_pred_labels = val_pred_logits.argmax(dim=1)
            # val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

def train(args,
          model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
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
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"Model saved with val loss: {best_val_loss:.4f}.")
            
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
        
def load_model_from_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    
    return model


def verify_gradient_flow(model):
    device = model.device
    num_classes = 1000
    batch_size = 2
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)

    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: [batch_size, num_classes]

    target = torch.randint(0, num_classes, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    loss.backward()

    print("\nGradient check:")
    all_have_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"No gradient for parameter: {name}")
                all_have_grad = False
            else:
                print(f"Gradient OK for parameter: {name}")
        else:
            print(f"Parameter does not require grad: {name}")

    if all_have_grad:
        print("\nAll parameters have gradients. Gradient flow is verified.")
    else:
        print("\nSome parameters do not have gradients. There may be an issue with gradient flow.")
        
        