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
import io
from torch.optim.optimizer import Optimizer
import os
import torch.nn as nn
import torch.nn.functional as F


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

class NTXentLoss_chat(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Cosine similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2N, 2N]
        sim = sim / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -float('inf'))

        # Positive indices: for i in [0, 2N), positive pair is at i + N (mod 2N)
        pos_idx = torch.arange(2 * batch_size, device=z.device)
        pos_pair_idx = (pos_idx + batch_size) % (2 * batch_size)

        # Compute loss
        loss = F.cross_entropy(sim, pos_pair_idx)
        return loss

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""
    def __init__(self, temperature=0.5, verbose=False):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.verbose = verbose

    def forward(self, z_i, z_j):
        verbose = self.verbose
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # Stack positive pairs
        similarity_matrix = torch.matmul(z, z.T)  # Cosine similarity
        #I don't normalize because the model already does it in the forward pass
        
        # Remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        if verbose:
            print("similarity_matrix: ",similarity_matrix.shape)
            print(similarity_matrix)
        
        # Compute positive pairs similarity
        '''
        positives = torch.cat([torch.diag(similarity_matrix, batch_size-1), 
                               torch.diag(similarity_matrix, -batch_size+1)], dim=0)
        '''
        
        # Compute NT-Xent loss
        #labels = torch.arange(2 * batch_size, device=z.device)
        labels = torch.cat([torch.arange(batch_size-1,2*batch_size-1, device=z.device),
                            torch.arange(batch_size, device=z.device)], dim=0)
        if verbose:
            print("labels: ",labels.shape)
            print(labels)
        
        # Each row should have the highest score at its label index to be used by the crossentropy loss
        loss = self.criterion(similarity_matrix / self.temperature, labels)
        #labels should be the class indexes. The first argument are the logits.
        return loss

def get_criterion(name='CrossEntropyLoss'):
    if name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif name == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    elif name == 'MSELoss':
        return torch.nn.MSELoss()
    elif name == 'NTXentLoss':
        return NTXentLoss_chat()
    else:
        raise ValueError(f"Unknown criterion name: {name}. Please provide a valid criterion name.")

class LARS(Optimizer):
    def __init__(self, params, lr, weight_decay=1e-6, momentum=0.9, eta=0.001, eps=1e-9):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, eps=eps)
        super(LARS, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                dp = p.grad.data

                if group['weight_decay'] != 0:
                    dp = dp.add(group['weight_decay'], p.data)

                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(dp)
                one = torch.ones_like(param_norm)

                q = torch.where(param_norm > 0,
                                torch.where(grad_norm > 0,
                                            (group['eta'] * param_norm / (grad_norm + group['eps'])),
                                            one),
                                one)

                dp = dp.mul(q)

                if group.get('momentum', 0) > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(dp).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(dp)
                    dp = buf

                p.data.add_(-group['lr'], dp)

def get_optimizer(parameters, name='Adam', lr=0.001,**kwargs):
    if name == 'Adam':
        if 'weight_decay' in kwargs:
            weight_decay = kwargs['weight_decay']
        else:
            weight_decay = 0
        return optim.Adam(parameters, lr=lr,weight_decay=weight_decay)
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

def get_cosine_schedule_with_warmup(optimizer,**kwargs):
    warmup_epochs = kwargs.get('warmup_epochs', 5)
    total_epochs = kwargs.get('total_epochs', 100)
    init_epoch = kwargs.get('init_epoch', 0)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def get_cosine_schedule_custom(optimizer,start_epoch,**kwargs):
    total_epochs = kwargs.get('T_max', 100)
    def lr_lambda(current_epoch):
        progress = float(current_epoch) / float(max(1, total_epochs - start_epoch))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def get_linear(optimizer,start_epoch,**kwargs):
    warmup_epochs = kwargs.get('warmup_epochs', 5)
    def lr_lambda(current_epoch):
        print(f"Current epoch: {current_epoch}, Start epoch: {start_epoch}, Warmup epochs: {warmup_epochs}")
        return float(current_epoch+1) / float(max(1, warmup_epochs- start_epoch+1))
    return LambdaLR(optimizer, lr_lambda)

def get_scheduler(optimizer, name='no_scheduling',start_epoch=0, **kwargs):
    if name == 'CosineAnnealingLR':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,last_epoch=start_epoch-1)
    elif name == 'StepLR':
        step_size= kwargs.get('step_size', 30)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif name == 'ReduceLROnPlateau':
        patience = kwargs.get('patience', 10)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
    elif name == "no_scheduling":
        class NoOpScheduler:
            def step(self, *args, **kwargs):
                pass
            def state_dict(self):
                return {}
            def load_state_dict(self, state_dict):
                pass
        return NoOpScheduler()
    elif name == 'CosineAnnealingWarmup':
        return get_cosine_schedule_with_warmup(optimizer, **kwargs)
    elif name == 'CosineScheduleCustom':
        return get_cosine_schedule_custom(optimizer, start_epoch, **kwargs)
    elif name == 'Linear':
        return get_linear(optimizer, start_epoch=start_epoch, **kwargs)
    elif name == 'CyclicalLR':
        base_lr = kwargs.get('base_lr_cycle', 0.001)
        max_lr = kwargs.get('max_lr_cycle', 0.01)
        step_size_up = kwargs.get('step_size_up', 2000)
        step_size_down = kwargs.get('step_size_down', 2000)
        mode = kwargs.get('mode', 'triangular') #"exp_range", "triangular2"
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, 
                                                  step_size_up=step_size_up, step_size_down=step_size_down, 
                                                  mode=mode, cycle_momentum=False)
    elif name == 'OneCycleLR':
        max_lr = kwargs.get('max_lr', 0.01)
        #total_steps = kwargs.get('total_epochs', 1000)
        epochs = kwargs.get('total_epochs', 100)
        steps_per_epoch = kwargs.get('steps_per_epoch', 705)
        pct_start = kwargs.get('pct_start', 0.3)
        anneal_strategy = kwargs.get('anneal_strategy', 'cos')
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                                                   pct_start=pct_start, anneal_strategy=anneal_strategy)
    elif name == 'CosineAnnealingWarmRestarts':
        T_0 = kwargs.get('T_0', 10)
        T_mult = kwargs.get('T_mult', 1)
        eta_min = kwargs.get('eta_min', 0)
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=start_epoch-1)
    else:
        raise ValueError(f"Unknown scheduler name: {name}. Please provide a valid scheduler name.")

def get_model_size_mb(model):
    """Calculate model size in MB"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / 1e6
    return size_mb

def perform_training_step(model, batch, optimizer, scaler, loss_fn, device, 
                         use_amp, log_grad_norm, profiler, contrastive_mode=False):
    """Perform a single training step"""
    #print(f"[DEBUG] contrastive_mode in perform_training_step = {contrastive_mode}, batch keys = {batch.keys()}")
    if contrastive_mode:
        x_i, x_j = batch['image1'].to(device), batch['image2'].to(device)
    else:
        x_i = batch['image'].to(device)
        labels = batch['label'].to(device)
    optimizer.zero_grad(set_to_none=True)
    
    grad_norm = 0
    
    with profiler.record_function("training_step"):
        if use_amp:
            with autocast(device_type='cuda'):
                with profiler.record_function("forward_pass"):
                    if contrastive_mode:
                        z_i, z_j = model(x_i), model(x_j)
                        loss = loss_fn(z_i, z_j)
                    else:
                        z_i = model(x_i)
                        loss = loss_fn(z_i, labels)
                
                with profiler.record_function("backward_pass"):
                    scaler.scale(loss).backward()

                if log_grad_norm:
                    with profiler.record_function("gradient_clipping"):
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                else:
                    grad_norm=get_grad_norm(model, norm_type=2)

                with profiler.record_function("optimizer_step"):
                    scaler.step(optimizer)
                    scaler.update()
        else:
            with profiler.record_function("forward_pass"):
                if contrastive_mode:
                    z_i, z_j = model(x_i), model(x_j)
                    loss = loss_fn(z_i, z_j)
                else:
                    z_i = model(x_i)
                    loss = loss_fn(z_i, labels)
            
            with profiler.record_function("backward_pass"):
                loss.backward()

            if log_grad_norm:
                with profiler.record_function("gradient_clipping"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            else:
                grad_norm = get_grad_norm(model, norm_type=2)

            with profiler.record_function("optimizer_step"):
                optimizer.step()
    if contrastive_mode:
        return loss.detach().item(), grad_norm, z_i.detach(), None
    else:
        return loss.detach().item(), grad_norm, z_i.detach(), labels.detach()


def train_epoch(model, train_dataloader, optimizer, scheduler, scaler, loss_fn, 
                device, epoch, total_epochs, use_amp, log_grad_norm, profiler, metrics, step_at_epoch=False, contrastive_mode=False):
    """Train for one epoch"""
    model.train()
    epoch_train_loss = 0
    epoch_grad_norms = []
    correct = 0
    total = 0
    
    # Memory tracking at epoch start
    memory_before = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1e6
    #print(f"[DEBUG] contrastive_mode in train_epoch = {contrastive_mode}")
    loop = enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]"))
    
    for idx, batch in loop:
        batch_start_time = time.time()
        #print(f"[DEBUG] contrastive_mode in train_epoch = {contrastive_mode}")
        loss, grad_norm, outputs, labels = perform_training_step(
            model, batch, optimizer, scaler, loss_fn, device, 
            use_amp, log_grad_norm, profiler, contrastive_mode=contrastive_mode
        )

        if contrastive_mode==False:
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        
        epoch_train_loss += loss
        epoch_grad_norms.append(grad_norm)
        
        profiler.step()
        if step_at_epoch:
            scheduler.step() 
        
        # Detailed timing and memory tracking for first few batches
        if idx < 5 or (idx % 100 == 0):
            batch_time = time.time() - batch_start_time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                current_memory = torch.cuda.memory_allocated() / 1e6
                metrics.update_timing_stats(epoch, idx, batch_time, current_memory)
    
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0
    if contrastive_mode==False:
        train_accuracy = correct / total if total > 0 else 0.0
    else: 
        train_accuracy = 0.0
    
    return avg_train_loss, avg_grad_norm, memory_before, train_accuracy

def train_fine(
    model,
    train_dataloader,
    val_dataloader,
    device,
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
    optim_config=None,
    save_backbone=False,
    step_at_epoch=False,
    contrastive_mode=False,
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
    #optimizer = get_optimizer(model.parameters(),name=optim_name, lr=base_lr,**kwargs)
    #scheduler = get_scheduler(optimizer, name=scheduler_name, **kwargs) 
    optimization_manager = OptimizationManager(model, optim_config, **kwargs)
    scaler = GradScaler(device='cuda') if use_amp else None
    
    # Initialize helper classes
    metrics = TrainingMetrics()
    checkpoint_manager = CheckpointManager(checkpoint_path)
    checkpoint_manager_best = CheckpointManager(checkpoint_path.replace('.pt', '_best.pt'),save_backbone=save_backbone)
    
    # Load checkpoint if exists
    start_epoch,optimizer_dict,scheduler_dict, prev_opt_phase = checkpoint_manager.load_checkpoint(
        model, scaler, metrics, device, use_amp
    )
    optimizer = optimization_manager.set_optimizer(start_epoch)
    scheduler = optimization_manager.set_scheduler(optimizer,start_epoch)
    if start_epoch > 0:
        optimizer.load_state_dict(optimizer_dict)
        scheduler.load_state_dict(scheduler_dict)
    
    # Training loop
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        
        opt_phase = optimization_manager.get_phase(epoch)
        print(f"\nEpoch {epoch} - Optimization Phase: {opt_phase}")
        if prev_opt_phase != opt_phase:
            print(f"Switching to optimization phase {opt_phase} at epoch {epoch}")
            prev_opt_phase = opt_phase
            optimizer = optimization_manager.set_optimizer(epoch) #not sure if it destroys the loaded optimizer
            scheduler = optimization_manager.set_scheduler(optimizer,epoch)
        # Start epoch profiling
        # Setup profiler
        profiler = TrainingProfiler(use_profiler, profiler_config)
        profiler.setup_profiler()
        profiler.start_epoch_profiling(epoch)
        
        # Training phase
        avg_train_loss, avg_grad_norm, memory_before, train_accuracy = train_epoch(
            model, train_dataloader, optimizer, scheduler, scaler, loss_fn,
            device, epoch, total_epochs, use_amp, log_grad_norm, profiler, metrics, step_at_epoch=step_at_epoch, contrastive_mode=contrastive_mode
        )
        
        # Stop training profiler
        profiler.stop_epoch_profiling(epoch)
        
        # Validation phase
        val_start_time = time.time()
        
        if contrastive_mode:
            val_loss, val_acc = perform_validation_contrastive(model, val_dataloader, device, val_percentage, 
                                                               epoch, total_epochs, use_amp, loss_fn)
        else:
            val_loss, val_acc = perform_validation(model, val_dataloader, device, val_percentage, 
                                                   epoch, total_epochs, use_amp, loss_fn)
        
        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time
        
        # Update metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics.update_train_metrics(avg_train_loss, current_lr, train_accuracy ,avg_grad_norm)
        metrics.update_val_metrics(val_loss, val_acc)
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / 1e6
            metrics.update_memory_stats(epoch, memory_before, memory_after, epoch_time, val_time)
        
        # Step scheduler
        if step_at_epoch==False:
            if optimization_manager.scheduling=='ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Enhanced logging
        print(f"Epoch {epoch+1}| Train Accuracy {train_accuracy:.4f}| Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Avg Grad Norm: {avg_grad_norm:.4f} | "
              f"Epoch Time: {epoch_time:.2f}s | Val Time: {val_time:.2f}s")
        
        # Check for improvement and save checkpoint
        if metrics.check_improvement(val_loss):
            checkpoint_manager_best.save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, metrics, use_amp
            )
            print(f"✅ Saved new best model at epoch {epoch+1}")
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, metrics, opt_phase,use_amp
        )
        print(f"⏳ No improvement for {metrics.epochs_without_improvement} epoch(s)")
        
        # Early stopping check
        if metrics.should_early_stop(early_stopping_patience):
            print(f"⛔ Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break
        
        # Plot metrics
        if (epoch + 1) % plot_every == 0 or (epoch + 1) == total_epochs:
            checkpoint_folder = os.path.dirname(checkpoint_path)
            #print(f"Checkpoint folder: {checkpoint_folder}")
            metrics.plot_metrics(epoch, checkpoint_folder)
        
        # Check if we've run enough epochs
        if epoch >= start_epoch + run_epochs-1:
            print("Trained for required epochs, stopping training.")
            break
    
    # Final summary and cleanup
    print("\n📊 Performance Summary:")
    metrics.print_summary()
    #profiler.cleanup()
    best_ind = len(metrics.val_losses)-metrics.epochs_without_improvement-1 
    print(best_ind)
    best_model_performance = {
        'best_val_loss': metrics.best_val_loss,
        'best_val_acc': metrics.val_accuracies[best_ind],
        'best_epoch': best_ind,
        'best_train_loss': metrics.train_losses[best_ind],
        'best_train_acc': metrics.train_accuracies[best_ind],
        'last_epoch': len(metrics.val_losses)-1,
        'last_val_loss': metrics.val_losses[-1],
        'last_val_acc': metrics.val_accuracies[-1],
        'last_train_loss': metrics.train_losses[-1],
        'last_train_acc': metrics.train_accuracies[-1],
    }
    return best_model_performance 


class OptimizationManager:
    def __init__(self,model, config, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.optimizer_phases = config.get('optimizer_phases', [])
        self.phase_layers_to_freeze = config.get('phase_layers_to_freeze', [])
        self.phase_scheduling = config.get('phase_scheduling', [])
        self.phase_optimizer = config.get('phase_optimizer', ['Adam'])
        self.phase_lr = config.get('phase_lr', [0.001])
        self.phase_scheduler_hyperparams = config.get('phase_scheduler_hyperparams', [{}])
        self.phase_optimizer_hyperparams = config.get('phase_optimizer_hyperparams', [{}])
    
    def get_phase(self,epoch):
        for i, phase in enumerate(self.optimizer_phases):
            if phase > epoch:
                return i
        return len(self.optimizer_phases) - 1

    def set_optimizer(self,epoch):
        self.epoch = epoch
        i = self.get_phase(epoch)
        self.optimizer_name = self.phase_optimizer[i]
        self.lr = self.phase_lr[i]
        self.layers_to_freeze = self.phase_layers_to_freeze[i]
        for name, param in self.model.named_parameters():
            if name in self.layers_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print(f"Setting optimizer for phase {i}: {self.optimizer_name} with learning rate {self.lr}")
        print(f"Freezing {self.layers_to_freeze} parameters")
        # Print number of trainable parameters after freezing
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters after freezing: {trainable_params:,}")
        return get_optimizer(self.model.parameters(), name=self.optimizer_name, lr=self.lr, **self.phase_optimizer_hyperparams[i])

    def set_scheduler(self, optimizer,epoch):
        i = self.get_phase(epoch)
        self.scheduling = self.phase_scheduling[i]
        return get_scheduler(optimizer, name=self.scheduling, start_epoch = epoch, **self.phase_scheduler_hyperparams[i])


def get_progressive_training_steps(selected_model,contrastive_mode=False):
    if selected_model == 'resnet18':
        steps=['layer4','layer3','layer2','layer1']
    elif selected_model == 'DeiT-Tiny':
        steps=[f'layer.{i}'for i in range(11,-1,-1)]
    if contrastive_mode:
        return ['projection_head'] + steps
    else:
        return steps

def get_grad_norm(model, norm_type=2):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]

    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class BaseSearchConfig:
    def get_params(self, type_of_training, type_of_model):
        raise NotImplementedError


class GridSearchConfig(BaseSearchConfig):
    def get_params(self, type_of_training, type_of_model):
        configs = {
            'progressive': {
                'CNN': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-4,0.001, 0.01, 0.1],
                    'lr_backbone_scaling':[0.01,0.1,0.5],
                    'pretrain_head_epochs': [3,5],
                    'step_phase': [4,8,16],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler': ['no_scheduling'], #fisso
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-5,1e-4,0.001, 0.01],
                    'lr_backbone_scaling':[0.01,0.1,0.5],
                    'pretrain_head_epochs': [3,5],
                    'step_phase': [4,8,16],
                    'weight_decay': [1e-2,0.5],
                    'scheduler': ['no_scheduling'], #fisso
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                }
            },
            'fine_tune': {
                'CNN': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-4,0.001, 0.01, 0.1],
                    'pretrain_head_epochs': [3,5],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler_head': ['no_scheduling','CosineAnnealingLR'], 
                    'scheduler': ['no_scheduling','OneCycleLR','CosineAnnealingLR'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-5,1e-4,0.001, 0.01],
                    'pretrain_head_epochs': [3,5],
                    'step_phase': [4,8,16],
                    'weight_decay': [1e-2,0.5],
                    'scheduler_head': ['no_scheduling','CosineAnnealingLR'], 
                    'scheduler': ['no_scheduling'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                }
            },
            'from_scratch': {
                'CNN': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_initial': [1e-4,0.001, 0.01, 0.1],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler': ['no_scheduling','OneCycleLR','CosineAnnealingLR'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_initial': [1e-4,0.001, 0.01, 0.1],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler': ['no_scheduling','OneCycleLR','CosineAnnealingLR'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                }
            },
        }

        try:
            return configs[type_of_training][type_of_model]
        except KeyError as e:
            raise ValueError(f"Invalid GridSearch config: {e}")


class RandomSearchConfig(BaseSearchConfig):
    def get_params(self, type_of_training, type_of_model):
        configs = {
            'progressive': {
                'CNN': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-4,0.001, 0.01, 0.1],
                    'lr_backbone_scaling':[0.01,0.1,0.5],
                    'pretrain_head_epochs': [3,5],
                    'step_phase': [4,8,16],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler': ['no_scheduling'], #fisso
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-5,1e-4,0.001, 0.01],
                    'lr_backbone_scaling':[0.01,0.1,0.5],
                    'pretrain_head_epochs': [3,5],
                    'step_phase': [4,8,16],
                    'weight_decay': [1e-2,0.5],
                    'scheduler': ['no_scheduling'], #fisso
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                }
            },
            'fine_tune': {
                'CNN': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-4,0.001, 0.01, 0.1],
                    'pretrain_head_epochs': [3,5],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler_head': ['no_scheduling','CosineAnnealingLR'], 
                    'scheduler': ['no_scheduling','OneCycleLR','CosineAnnealingLR'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_classific_head': [0.001, 0.01, 0.1],
                    'lr_backbone_initial': [1e-5,1e-4,0.001, 0.01],
                    'pretrain_head_epochs': [3,5],
                    'step_phase': [4,8,16],
                    'weight_decay': [1e-2,0.5],
                    'scheduler_head': ['no_scheduling','CosineAnnealingLR'], 
                    'scheduler': ['no_scheduling'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                }
            },
            'from_scratch': {
                'CNN': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_initial': [1e-4,0.001, 0.01, 0.1],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler': ['no_scheduling','OneCycleLR','CosineAnnealingLR'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1','MLPClassifier2'],
                    'optimizer': ['AdamW','Adam','SGD'],
                    'lr_initial': [1e-4,0.001, 0.01, 0.1],
                    'weight_decay': [1e-5, 1e-4,1e-3],
                    'scheduler': ['no_scheduling','OneCycleLR','CosineAnnealingLR'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2, 0.4],
                    'n_neurons': [128, 256],
                    'with_input_norm': [True,False],
                }
            },
        }

        try:
            return configs[type_of_training][type_of_model]
        except KeyError as e:
            raise ValueError(f"Invalid RandomSearch config: {e}")

class SingleSearchConfig(BaseSearchConfig):
    def get_params(self, type_of_training, type_of_model):
        configs = {
            'progressive': {
                'CNN': {
                    #https://chatgpt.com/share/689dff31-5c94-8010-88e7-08dd47b8f061
                    'head_model': ['MLPClassifier1'],
                    'optimizer': ['AdamW'],
                    'lr_classific_head': [1e-4],
                    'lr_backbone_initial': [1e-6],
                    'lr_backbone_scaling':[0.1],
                    'pretrain_head_epochs': [3],
                    'step_phase': [3],
                    'weight_decay': [5e-4],
                    'scheduler': ['no_scheduling'], #fisso
                    'log_grad_norm': [True],
                    'dropout': [0.2],
                    'n_neurons': [128],
                    'with_input_norm': ['batch_norm'],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1'],
                    'optimizer': ['AdamW'],
                    'lr_classific_head': [1e-4],
                    'lr_backbone_initial': [1e-6],
                    'lr_backbone_scaling':[0.1],
                    'pretrain_head_epochs': [3],
                    'step_phase': [5],
                    'weight_decay': [1e-4],
                    'scheduler': ['no_scheduling'], #fisso
                    'log_grad_norm': [False],
                    'dropout': [0.2],
                    'n_neurons': [128],
                    'with_input_norm': ['batch_norm']
                },
            },
            'fine_tune': {
                'CNN': {
                    'head_model': ['MLPClassifier1'],
                    'optimizer': ['Adam'],
                    'lr_classific_head': [1e-3],
                    'lr_backbone_initial': [1e-4],
                    'pretrain_head_epochs': [3],
                    'weight_decay': [1e-4],
                    'scheduler_head': ['no_scheduling'], 
                    'scheduler': ['CosineAnnealingLR'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2],
                    'n_neurons': [64],
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1'],
                    'optimizer': ['AdamW'],
                    'lr_classific_head': [0.001],
                    'lr_backbone_initial': [1e-4],
                    'pretrain_head_epochs': [3],
                    'weight_decay': [1e-5],
                    'scheduler_head': ['no_scheduling'], 
                    'scheduler': ['no_scheduling'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2],
                    'n_neurons': [128],
                    'with_input_norm': [True]
                }
            },
            'from_scratch': {
                'CNN': {
                    'head_model': ['MLPClassifier1'],
                    'optimizer': ['AdamW'],
                    'lr_initial': [1e-4],
                    'weight_decay': [1e-5],
                    'scheduler': ['no_scheduling'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2],
                    'n_neurons': [128],
                    'with_input_norm': [True]
                },
                'Transformer': {
                    'head_model': ['MLPClassifier1'],
                    'optimizer': ['AdamW'],
                    'lr_initial': [1e-4],
                    'weight_decay': [1e-5],
                    'scheduler': ['no_scheduling'], 
                    'log_grad_norm': [False],
                    'dropout': [0.2],
                    'n_neurons': [128],
                    'with_input_norm': [True]
                }
            },
        }

        try:
            return configs[type_of_training][type_of_model]
        except KeyError as e:
            raise ValueError(f"Invalid SingleSearch config: {e}")

class SearchConfigFactory:
    def __init__(self):
        self._registry = {
            'grid_search': GridSearchConfig(),
            'random_search': RandomSearchConfig(),
            'single_search': SingleSearchConfig(),
        }

    def get_config(self, type_of_search):
        try:
            return self._registry[type_of_search]
        except KeyError:
            raise ValueError(f"Unknown search type: {type_of_search}")


def get_search_params(type_of_search='grid_search', type_of_training='progressive', type_of_model='CNN'):
    factory = SearchConfigFactory()
    config = factory.get_config(type_of_search)
    return config.get_params(type_of_training, type_of_model)

def set_search_params(params,type_of_training,total_epochs,steps,backbone_param_names):
    """Set search parameters based on the type of training"""
    if type_of_training == 'progressive':
        #for progressive fine tuning
        optimizer_phases = [params['pretrain_head_epochs']]
        phase_layers_to_freeze = [backbone_param_names]
        phase_lr = [params['lr_classific_head']]
        step_phase = params['step_phase']
        lr_backbone_initial = params['lr_backbone_initial']
        lr_backbone_scaling = params['lr_backbone_scaling']
        for i,name in enumerate(steps):
            optimizer_phases.append(optimizer_phases[i]+step_phase) 
            phase_layers=phase_layers_to_freeze[i]
            phase_layers_to_freeze.append([l for l in phase_layers if not(name in l)])
            if i == 0:
                phase_lr.append(lr_backbone_initial)
            else:
                phase_lr.append(phase_lr[i] * lr_backbone_scaling)  # Decrease learning rate for each phase
        phase_lr += [phase_lr[-1] * lr_backbone_scaling]  # Final phase learning rate
        optimizer_name = params['optimizer']
        weight_decay = params['weight_decay'] 
        optim_config = {
                'optimizer_phases':optimizer_phases+[total_epochs],  # Example: [10, 10, 80] for 100 epochs
                'phase_layers_to_freeze':phase_layers_to_freeze+[[]],
                'phase_scheduling': [params['scheduler'] for _ in range(len(optimizer_phases)+1)],
                'phase_optimizer':[optimizer_name for _ in range(len(optimizer_phases)+1)],  # Example: ['AdamW', 'SGD', 'AdamW'] for different phases
                'phase_lr': phase_lr,
                'phase_optimizer_hyperparams': [{'weight_decay':weight_decay} for _ in range(len(optimizer_phases)+1) for _ in range(len(optimizer_phases)+1)],
                'phase_scheduler_hyperparams': [{} for _ in range(len(optimizer_phases)+1)],
            }
    elif type_of_training == 'fine_tune':
        optimizer_phases = [params['pretrain_head_epochs'], total_epochs]  # Example: [1, 4, 95] for 100 epochs
        optim_config = {
            'optimizer_phases':optimizer_phases,  # Example: [10, 10, 80] for 100 epochs
            'phase_layers_to_freeze':[backbone_param_names,[]],
            'phase_scheduling': [params['scheduler_head'],params['scheduler']],
            'phase_optimizer':[params['optimizer'],params['optimizer']],  # Example: ['AdamW', 'SGD', 'AdamW'] for different phases
            'phase_lr': [params['lr_classific_head'], params['lr_backbone_initial']],
            'phase_optimizer_hyperparams': [{'weight_decay': params['weight_decay'] }for i in range(2)],
            'phase_scheduler_hyperparams': [{'warmup_epochs': optimizer_phases[0], 'T_max': total_epochs} for i in range(2)],
        }
    elif type_of_training == 'from_scratch':
        optimizer_phases = [total_epochs]  # Example: [1, 4, 95] for 100 epochs
        optim_config = {
            'optimizer_phases':optimizer_phases,  # Example: [10, 10, 80] for 100 epochs
            'phase_layers_to_freeze':[[]],
            'phase_scheduling': [params['scheduler']],
            'phase_optimizer':[params['optimizer']],  # Example: ['AdamW', 'SGD', 'AdamW'] for different phases
            'phase_lr': [params['lr_initial']],
            'phase_optimizer_hyperparams': [{'weight_decay': weight_decay}],
            'phase_scheduler_hyperparams': [{'warmup_epochs': optimizer_phases[0], 'T_max': total_epochs}],
        }
    else:
        raise ValueError(f"Unknown training type: {type_of_training}")
    return optim_config

