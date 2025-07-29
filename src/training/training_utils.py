"""
Training utilities and helper classes.
"""

import os
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """

    def __init__(self,
                 patience: int = 7,
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change in monitored quantity to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            mode: 'min' or 'max' - whether to minimize or maximize metric
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            score: Current validation score
            model: Model to save best weights (if restore_best_weights=True)

        Returns:
            True if early stopping should be triggered
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                if model is not None and self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info('Restored best model weights')

        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:  # mode == 'max'
            return score > self.best_score + self.min_delta


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """

    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 verbose: bool = True):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Path to save checkpoint (can include formatting placeholders)
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' - whether to minimize or maximize metric
            save_best_only: Whether to save only when metric improves
            save_weights_only: Whether to save only model weights
            verbose: Whether to print checkpoint messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best_score = None

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def __call__(self, model: nn.Module, current_score: float, epoch: int,
                 additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save checkpoint if criteria are met.

        Args:
            model: Model to save
            current_score: Current metric value
            epoch: Current epoch number
            additional_info: Additional information to save
        """
        if self.save_best_only:
            if self.best_score is None or self._is_improvement(current_score):
                self.best_score = current_score
                self._save_checkpoint(model, epoch, additional_info)
                if self.verbose:
                    logger.info(f'Checkpoint saved at epoch {epoch} with {self.monitor}={current_score:.4f}')
        else:
            self._save_checkpoint(model, epoch, additional_info)
            if self.verbose:
                logger.info(f'Checkpoint saved at epoch {epoch}')

    def _is_improvement(self, score: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score
        else:  # mode == 'max'
            return score > self.best_score

    def _save_checkpoint(self, model: nn.Module, epoch: int,
                        additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Save the checkpoint."""
        # Format filepath with epoch number
        filepath = self.filepath.format(epoch=epoch)

        if self.save_weights_only:
            torch.save(model.state_dict(), filepath)
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best_score
            }

            if additional_info:
                checkpoint.update(additional_info)

            torch.save(checkpoint, filepath)


class TrainingLogger:
    """
    Logger for training metrics and progress.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs (None for no file logging)
        """
        self.log_dir = log_dir
        self.metrics_history = []

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, 'training_log.jsonl')

    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        log_entry = {'epoch': epoch, 'timestamp': time.time(), **metrics}
        self.metrics_history.append(log_entry)

        # Save to file if log_dir is specified
        if self.log_dir:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    def plot_metrics(self, metrics: List[str], save_path: Optional[str] = None) -> None:
        """
        Plot training metrics.

        Args:
            metrics: List of metric names to plot
            save_path: Path to save plot (None to display)
        """
        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return

        epochs = [entry['epoch'] for entry in self.metrics_history]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            values = [entry.get(metric, 0) for entry in self.metrics_history]
            axes[i].plot(epochs, values, label=metric)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} over time')
            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()

    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics history to JSON file.

        Args:
            filepath: Path to save metrics
        """
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class GradientAccumulator:
    """
    Helper class for gradient accumulation.
    """

    def __init__(self, accumulate_steps: int):
        """
        Initialize gradient accumulator.

        Args:
            accumulate_steps: Number of steps to accumulate gradients
        """
        self.accumulate_steps = accumulate_steps
        self.current_step = 0

    def step(self) -> bool:
        """
        Take a step and return whether to update optimizer.

        Returns:
            True if optimizer should be updated
        """
        self.current_step += 1
        return self.current_step % self.accumulate_steps == 0

    def reset(self) -> None:
        """Reset the accumulator."""
        self.current_step = 0


class MetricTracker:
    """
    Track and compute running averages of metrics.
    """

    def __init__(self):
        """Initialize metric tracker."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1) -> None:
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric values
            count: Number of samples (for weighted average)
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value * count
            self.counts[key] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.

        Returns:
            Dictionary of averaged metrics
        """
        return {key: self.metrics[key] / self.counts[key]
               for key in self.metrics.keys()}

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get appropriate device for training.

    Args:
        device_id: CUDA device ID (None for auto-select)

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


def save_training_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save training configuration to file.

    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)  # Test if serializable
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)

    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_training_config(filepath: str) -> Dict[str, Any]:
    """
    Load training configuration from file.

    Args:
        filepath: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def epoch_time(start_time: float, end_time: float) -> tuple:
    """
    Calculate epoch time in minutes and seconds.

    Args:
        start_time: Start time timestamp
        end_time: End time timestamp

    Returns:
        Tuple of (minutes, seconds)
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Get information about optimizer state.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Dictionary with optimizer information
    """
    info = {
        'optimizer_type': type(optimizer).__name__,
        'num_param_groups': len(optimizer.param_groups),
        'learning_rates': [group['lr'] for group in optimizer.param_groups]
    }

    # Add optimizer-specific parameters
    if hasattr(optimizer, 'defaults'):
        info['defaults'] = optimizer.defaults

    return info


def freeze_model_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specific layers in a model.

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")


def unfreeze_model_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Unfreeze specific layers in a model.

    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory usage information
    """
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3
        }
    else:
        return {'message': 'CUDA not available'}


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"