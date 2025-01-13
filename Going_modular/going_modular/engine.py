
import torch
from torch import nn
from tqdm.auto import tqdm
# from timeit import default_timer as timer
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()

    # Setup train loss, train accuracy
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (x, y) in enumerate(dataloader):
      # Send x, y to device
      x, y = x.to(device), y.to(device)

      # Forward
      y_pred = model(x)

      # Calculate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item()

      # Optimizer zero grad
      optimizer.zero_grad()

      # Loss backward
      loss.backward()

      # Optimizer step
      optimizer.step()

      # Calculate accuracy
      y_pred_class = torch.argmax(y_pred, dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc



def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  # Put model in eval mode
  model.eval()

  # Set up test loss, test accuracy
  test_loss, test_acc = 0, 0

  # Turn on inference mode
  with torch.inference_mode():
    # Loop through data loader batches
    for batch, (x, y) in enumerate(dataloader):
      # Send x, y to device
      x, y = x.to(device), y.to(device)

      # Forwardind
      y_logits = model(x)

      # Calculate and accumulate loss
      loss = loss_fn(y_logits, y)
      test_loss += loss.item()

      # Calculate and accumulate accuracy
      y_pred = torch.argmax(y_logits, dim=1)
      test_acc += (y_pred == y).sum().item()/len(y_pred)

  # Calculate loss and accuracy per batch
  test_loss /= len(dataloader)
  test_acc /= len(dataloader)
  return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, list]:
  # Create empty results dictionary
  results ={
      'train_loss': [],
      'train_acc': [],
      'test_loss': [],
      'test_acc': []
  }

  # Loop through training and testing steps for numbers of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)
    # Print result per epoch
    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

    # Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
