from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
import torch.optim as optim
from datetime import datetime

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=5, checkpoint_path=None,early_stopping_patience=10, scheduler=None
                ,data_type='image'):
    start_time=datetime.now()
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        # Training Loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch[data_type], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = correct / total
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation Loop
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[data_type], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint if loss improves
        if checkpoint_path and avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
                'val_acc': val_acc,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'epoch': epoch,
                'time_from_start': datetime.now()-start_time,
            }
            torch.save(checkpoint, checkpoint_path+'best_checkpoint.pth')
            print(f"Checkpoint saved: {checkpoint_path}"+'best_checkpoint.pth')
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
                'val_acc': val_acc,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'epoch': epoch,
                'time_from_start': datetime.now()-start_time,
            }
            torch.save(checkpoint, checkpoint_path+'last_checkpoint.pth')
            print(f"Checkpoint saved: {checkpoint_path}"+'last_checkpoint.pth')
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
        if scheduler:
            scheduler.step()
    
    return model, train_losses, val_losses

#function to get all model layers without repeating layers
def get_all_layers(model):
    layers = []
    for layer in model.children():
        if list(layer.children()):  # If the layer has sub-layers, recurse
            layers.extend(get_all_layers(layer))
        else:
            layers.append(layer)
    return layers

# Assume 'model' is a pre-trained PyTorch model
def fine_tune_last_n_layers(model, n):
    # Print total number of layers of the model
    num_layers = len(list(model.parameters()))
    print(f"Total Layers: {num_layers}")

    # Print total number of parameters of the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Trainable parameters after freezing all layers
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters after freezing: {trainable_params:,}")

    # Unfreeze the last n layers
    layers = get_all_layers(model)
    if len(layers) < n:
        raise ValueError(f"Model has only {len(layers)} layers, but {n} were requested for fine-tuning.")
    print("\n")
    print("Unfreezing the following layers:")
    if n>0:
        for layer in layers[-n:]:  # Unfreeze the last n layers
            print(layer)
            for param in layer.parameters():
                param.requires_grad = True
    else:
        for layer in layers:  # Unfreeze all layers
            print(layer)
            for param in layer.parameters():
                param.requires_grad = True

    print("\n")
    # Trainable parameters after unfreezing 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters after UN-freezing last n layers: {trainable_params:,}")

    return model

def get_criterion(name='CrossEntropyLoss'):
    if name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif name == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    elif name == 'MSELoss':
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown criterion name: {name}. Please provide a valid criterion name.")
def get_optimizer(model, name='Adam', lr=0.001):
    if name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {name}. Please provide a valid optimizer name.")
def get_scheduler(optimizer, name='no_scheduling', T_max=10, eta_min=0):
    if name == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif name == "no_scheduling":
        return None
    else:
        raise ValueError(f"Unknown scheduler name: {name}. Please provide a valid scheduler name.")