import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Subset

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['image'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")


def perform_validation(model, val_dataloader, device, val_percentage, epoch, total_epochs, use_amp, loss_fn):
    """Perform validation using linear evaluation"""
    model.eval()
    reps, labels_list = [], []
    
    val_dataset = val_dataloader.dataset
    n_val = max(1, int(len(val_dataset) * val_percentage))
    subset_indices = random.sample(range(len(val_dataset)), n_val)

    subset = Subset(val_dataset, subset_indices)
    selected_val_batches = DataLoader(
        subset,
        batch_size=val_dataloader.batch_size,
        shuffle=False,
        num_workers=val_dataloader.num_workers,
        pin_memory=val_dataloader.pin_memory,
    )
    
    # Extract representations
    val_loss = 0.0
    correct = 0
    total = 0

    if use_amp:
        with torch.no_grad(), autocast(device_type='cuda'):
            for batch in tqdm(selected_val_batches, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                inputs, labels = batch['image'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
    else:
        with torch.no_grad():
            for batch in tqdm(selected_val_batches, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                inputs, labels = batch['image'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

    val_loss = val_loss / total if total > 0 else 0
    val_acc = correct / total if total > 0 else 0
    

    return val_loss, val_acc

