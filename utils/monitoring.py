import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

class TrainingProfiler:
    """Handles all profiling-related functionality"""
    
    def __init__(self, use_profiler=False, config=None):
        self.use_profiler = use_profiler
        self.config = self._get_default_config() if config is None else config
        self.profiler = None
        self.current_epoch_profiling = False
        self.traces_exported = False  # <-- Add this
        
    def _get_default_config(self):
        return {
            'profile_epochs': [1, 2],
            'profile_batches': 20,
            'output_dir': './profiler_logs',
            'profile_memory': True,
            'profile_shapes': True,
            'with_stack': True,
            'with_flops': True,
            'export_chrome_trace': True,
            'export_stacks': True,
        }
    
    def setup_profiler(self):
        """Initialize the profiler if needed"""
        if not self.use_profiler:
            return
            
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=2, #2
                warmup=2, # 2
                active=self.config['profile_batches'],# self.config['profile_batches'],
                repeat=1 #1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.config['output_dir']
            ),
            record_shapes=self.config['profile_shapes'],
            profile_memory=self.config['profile_memory'],
            with_stack=self.config['with_stack'],
            with_flops=self.config['with_flops'],
        )
        #print(f"Profiler initialized with output directory: {self.config['output_dir']}")
    
    def should_profile_epoch(self, epoch):
        """Check if we should profile this epoch"""
        return (self.use_profiler and 
                epoch in self.config.get('profile_epochs', []))
    
    def start_epoch_profiling(self, epoch):
        """Start profiling for an epoch if configured"""
        self.current_epoch_profiling = self.should_profile_epoch(epoch)
        if self.current_epoch_profiling:
            print(f"🔍 Profiling enabled for epoch {epoch+1}")
            self.traces_exported = False  # Reset flag for new profiling session
            self.profiler.start()
    
    def stop_epoch_profiling(self, epoch):
        if not self.current_epoch_profiling or self.traces_exported:
            return

        self.profiler.stop()
        print(f"✅ Profiling completed for epoch {epoch+1}")

        # Only export if you did NOT use on_trace_ready handler
        # if self.config.get('export_chrome_trace', False):
        #     trace_path = os.path.join(self.config['output_dir'], f'trace_epoch_{epoch+1}.json')
        #     self.profiler.export_chrome_trace(trace_path)
        #     print(f"📊 Chrome trace exported to: {trace_path}")

        # if self.config.get('export_stacks', False):
        #     stacks_path = os.path.join(self.config['output_dir'], f'stacks_epoch_{epoch+1}.txt')
        #     self.profiler.export_stacks(stacks_path, "self_cuda_time_total")
        #     print(f"📊 Stack traces exported to: {stacks_path}")

        self.traces_exported = True
        self.profiler = None
    
    def step(self):
        """Step the profiler"""
        if self.current_epoch_profiling and self.profiler:
            #print("[Profiler] Stepping...")
            self.profiler.step()
    
    def record_function(self, name):
        """Return profiling context or nullcontext"""
        if self.current_epoch_profiling:
            return torch.profiler.record_function(name)
        return nullcontext()
    
    def cleanup(self):
        """Clean up profiler resources"""
        if self.profiler:
            self.profiler.stop()

class TrainingMetrics:
    """Handles tracking and visualization of training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.grad_norms = []
        self.memory_stats = []
        self.timing_stats = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def update_train_metrics(self, loss, lr, grad_norm=0):
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
    
    def update_val_metrics(self, loss, accuracy):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
    
    def update_memory_stats(self, epoch, memory_before, memory_after, epoch_time, val_time):
        self.memory_stats.append({
            'epoch': epoch,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_after - memory_before,
            'epoch_time': epoch_time,
            'val_time': val_time
        })
    
    def update_timing_stats(self, epoch, batch, batch_time, memory_mb):
        self.timing_stats.append({
            'epoch': epoch,
            'batch': batch,
            'batch_time': batch_time,
            'memory_mb': memory_mb
        })
    
    def check_improvement(self, val_loss):
        """Check if validation loss improved and update tracking"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def should_early_stop(self, patience):
        return self.epochs_without_improvement >= patience
    
    def plot_metrics(self, epoch, save_path=None):
        """Create comprehensive training plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.train_losses, label="Train Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].set_title("Training Loss")

        axes[0, 1].plot(self.val_losses, label="Val Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].set_title("Validation Loss")

        # Learning rate and grad norm
        axes[0, 2].plot(self.learning_rates, label="Learning Rate", color='orange',linestyle='None', marker='o')
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("LR")
        axes[0, 2].legend()
        axes[0, 2].set_title("Learning Rate")

        axes[1, 0].plot(self.grad_norms, label="Avg Grad Norm", color='green')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Grad Norm")
        axes[1, 0].legend()
        axes[1, 0].set_title("Gradient Norm")
        
        # Memory and timing plots
        if self.memory_stats:
            epochs_recorded = [s['epoch'] for s in self.memory_stats]
            memory_usage = [s['memory_after_mb'] for s in self.memory_stats]
            epoch_times = [s['epoch_time'] for s in self.memory_stats]
            
            axes[1, 1].plot(epochs_recorded, memory_usage, label="Memory Usage", color='red')
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Memory (MB)")
            axes[1, 1].legend()
            axes[1, 1].set_title("GPU Memory Usage")
            
            axes[1, 2].plot(epochs_recorded, epoch_times, label="Epoch Time", color='purple')
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("Time (s)")
            axes[1, 2].legend()
            axes[1, 2].set_title("Epoch Training Time")

        plt.suptitle(f"Training Progress - Epoch {epoch+1}")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "training_plot.png"))
        plt.close()
    
    def print_summary(self):
        """Print final training summary"""
        if self.timing_stats:
            avg_batch_time = np.mean([s['batch_time'] for s in self.timing_stats])
            print(f"Average batch time: {avg_batch_time:.4f}s")
            
        if self.memory_stats:
            max_memory = max([s['memory_after_mb'] for s in self.memory_stats])
            print(f"Peak GPU memory usage: {max_memory:.2f} MB")

class CheckpointManager:
    """Handles saving and loading of training checkpoints"""
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
    
    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, metrics, opt_phase,use_amp=True):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': metrics.best_val_loss,
            'train_losses': metrics.train_losses,
            'val_losses': metrics.val_losses,
            'val_accuracies': metrics.val_accuracies,
            'learning_rates': metrics.learning_rates,
            'grad_norms': metrics.grad_norms,
            'memory_stats': metrics.memory_stats,
            'timing_stats': metrics.timing_stats,
            'opt_phase': opt_phase,
        }
        
        if use_amp and scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
            
        torch.save(checkpoint, self.checkpoint_path)
    
    def load_checkpoint(self, model, scaler, metrics, device, use_amp=True):
        """Load training checkpoint and return start epoch"""
        if not os.path.exists(self.checkpoint_path):
            print(f"📂 No checkpoint found at {self.checkpoint_path}. Starting fresh training.")
            return 0, None, None, 0 
        
        print(f"🔄 Resuming from checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if use_amp and scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore metrics
        metrics.train_losses = checkpoint.get('train_losses', [])
        metrics.val_losses = checkpoint.get('val_losses', [])
        metrics.val_accuracies = checkpoint.get('val_accuracies', [])
        metrics.learning_rates = checkpoint.get('learning_rates', [])
        metrics.grad_norms = checkpoint.get('grad_norms', [])
        metrics.memory_stats = checkpoint.get('memory_stats', [])
        metrics.timing_stats = checkpoint.get('timing_stats', [])
        metrics.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch'],checkpoint['optimizer_state_dict'],checkpoint['scheduler_state_dict'],checkpoint['opt_phase']