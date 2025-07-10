from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
import torch.optim as optim
from datetime import datetime
from utils.monitoring import TrainingProfiler,TrainingMetrics, CheckpointManager
from utils.evaluation_utils import perform_validation
import time
import math
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

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

def get_optimizer(parameters, name='Adam', lr=0.001,**kwargs):
    if name == 'Adam':
        return optim.Adam(parameters, lr=lr)
    elif name == 'SGD':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    elif name == 'AdamW':
        if 'weight_decay' in kwargs:
            weight_decay = kwargs['weight_decay']
        else:
            weight_decay = 0.05
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer name: {name}. Please provide a valid optimizer name.")

def get_cosine_schedule_with_warmup(optimizer):
    warmup_epochs = kwargs.get('warmup_epochs', 5)
    total_epochs = kwargs.get('total_epochs', 100)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def get_scheduler(optimizer, name='no_scheduling', **kwargs):
    if name == 'CosineAnnealingLR':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif name == "no_scheduling":
        return None
    elif name == 'CosineAnnealingWarmRestarts':
        return get_cosine_schedule_with_warmup(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler name: {name}. Please provide a valid scheduler name.")

def get_model_size_mb(model):
    """Calculate model size in MB"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / 1e6
    return size_mb

def perform_training_step(model, batch, optimizer, scaler, loss_fn, device, 
                         use_amp, log_grad_norm, profiler):
    """Perform a single training step"""
    x_i = batch['image'].to(device)
    labels = batch['label'].to(device)
    optimizer.zero_grad(set_to_none=True)
    
    grad_norm = 0
    
    with profiler.record_function("training_step"):
        if use_amp:
            with autocast(device_type='cuda'):
                with profiler.record_function("forward_pass"):
                    z_i = model(x_i)
                    loss = loss_fn(z_i, labels)
                
                with profiler.record_function("backward_pass"):
                    scaler.scale(loss).backward()

                if log_grad_norm:
                    with profiler.record_function("gradient_clipping"):
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

                with profiler.record_function("optimizer_step"):
                    scaler.step(optimizer)
                    scaler.update()
        else:
            with profiler.record_function("forward_pass"):
                z_i = model(x_i)
                loss = loss_fn(z_i, labels)
            
            with profiler.record_function("backward_pass"):
                loss.backward()

            if log_grad_norm:
                with profiler.record_function("gradient_clipping"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            with profiler.record_function("optimizer_step"):
                optimizer.step()
    
    return loss.detach().item(), grad_norm

def train_epoch(model, train_dataloader, optimizer, scheduler, scaler, loss_fn, 
                device, epoch, total_epochs, use_amp, log_grad_norm, profiler, metrics):
    """Train for one epoch"""
    model.train()
    epoch_train_loss = 0
    epoch_grad_norms = []
    
    # Memory tracking at epoch start
    memory_before = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1e6
    
    loop = enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]"))
    
    for idx, batch in loop:
        batch_start_time = time.time()
        
        loss, grad_norm = perform_training_step(
            model, batch, optimizer, scaler, loss_fn, device, 
            use_amp, log_grad_norm, profiler
        )
        
        epoch_train_loss += loss
        if log_grad_norm:
            epoch_grad_norms.append(grad_norm)
        
        profiler.step()
        
        # Detailed timing and memory tracking for first few batches
        if idx < 5 or (idx % 100 == 0):
            batch_time = time.time() - batch_start_time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                current_memory = torch.cuda.memory_allocated() / 1e6
                metrics.update_timing_stats(epoch, idx, batch_time, current_memory)
    
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0
    
    return avg_train_loss, avg_grad_norm, memory_before

def train_fine(
    model,
    train_dataloader,
    val_dataloader,
    device,
    base_lr,
    warmup_epochs=10,
    total_epochs=100,
    checkpoint_path='checkpoint.pt',
    plot_every=5,
    early_stopping_patience=10, 
    loss_fn=None,  
    val_percentage=0.1,
    use_profiler=False,
    profiler_config=None,
    log_grad_norm=False,
    run_epochs=2,
    save_path=None,
    use_amp=True,
    scheduler_name=None,
    optim_name='Adam',
    **kwargs
):
    """
    Main training function - now much cleaner and more readable
    """
    # Setup components
    model = model.to(device)
    model_size_mb = get_model_size_mb(model)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Initialize components
    optimizer = get_optimizer(model.parameters(),name=optim_name, lr=base_lr,**kwargs)
    scheduler = get_scheduler(optimizer, name=scheduler_name, **kwargs) 
    scaler = GradScaler(device='cuda') if use_amp else None
    
    # Initialize helper classes
    metrics = TrainingMetrics()
    checkpoint_manager = CheckpointManager(checkpoint_path)
    checkpoint_manager_best = CheckpointManager(checkpoint_path.replace('.pt', '_best.pt'))
    
    # Load checkpoint if exists
    start_epoch = checkpoint_manager.load_checkpoint(
        model, optimizer, scheduler, scaler, metrics, device, use_amp
    )
    
    # Training loop
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        
        # Start epoch profiling
        # Setup profiler
        profiler = TrainingProfiler(use_profiler, profiler_config)
        profiler.setup_profiler()
        profiler.start_epoch_profiling(epoch)
        
        # Training phase
        avg_train_loss, avg_grad_norm, memory_before = train_epoch(
            model, train_dataloader, optimizer, scheduler, scaler, loss_fn,
            device, epoch, total_epochs, use_amp, log_grad_norm, profiler, metrics
        )
        
        # Stop training profiler
        profiler.stop_epoch_profiling(epoch)
        
        # Validation phase
        val_start_time = time.time()
        
        val_loss, val_acc = perform_validation(
            model, val_dataloader, device, val_percentage, epoch, total_epochs, use_amp, loss_fn
        )
        
        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time
        
        # Update metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics.update_train_metrics(avg_train_loss, current_lr, avg_grad_norm)
        metrics.update_val_metrics(val_loss, val_acc)
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / 1e6
            metrics.update_memory_stats(epoch, memory_before, memory_after, epoch_time, val_time)
        
        # Step scheduler
        scheduler.step()
        
        # Enhanced logging
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Avg Grad Norm: {avg_grad_norm:.4f} | "
              f"Epoch Time: {epoch_time:.2f}s | Val Time: {val_time:.2f}s")
        
        # Check for improvement and save checkpoint
        if metrics.check_improvement(val_loss):
            checkpoint_manager_best.save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, metrics, use_amp
            )
            print(f"✅ Saved new best model at epoch {epoch+1}")
        else:
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, metrics, use_amp
            )
            print(f"⏳ No improvement for {metrics.epochs_without_improvement} epoch(s)")
        
        # Early stopping check
        if metrics.should_early_stop(early_stopping_patience):
            print(f"⛔ Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break
        
        # Plot metrics
        if (epoch + 1) % plot_every == 0 or (epoch + 1) == total_epochs:
            metrics.plot_metrics(epoch, save_path)
        
        # Check if we've run enough epochs
        if epoch >= start_epoch + run_epochs-1:
            print("Trained for required epochs, stopping training.")
            break
    
    # Final summary and cleanup
    print("\n📊 Performance Summary:")
    metrics.print_summary()
    #profiler.cleanup()
