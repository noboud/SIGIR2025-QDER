"""
Learning rate schedulers for QDER model training.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional, Union
import warnings


class WarmupLinearSchedule(_LRScheduler):
    """
    Linear warmup followed by linear decay.

    Commonly used in transformer training.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 last_epoch: int = -1):
        """
        Initialize warmup linear scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay phase
            decay_factor = max(0.0, float(self.total_steps - self.last_epoch) /
                               float(max(1, self.total_steps - self.warmup_steps)))
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupCosineSchedule(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr_ratio: float = 0.0,
                 last_epoch: int = -1):
        """
        Initialize warmup cosine scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate as ratio of base LR
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
                    for base_lr in self.base_lrs]


class PolynomialDecaySchedule(_LRScheduler):
    """
    Polynomial decay learning rate schedule.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 total_steps: int,
                 power: float = 1.0,
                 min_lr_ratio: float = 0.0,
                 last_epoch: int = -1):
        """
        Initialize polynomial decay scheduler.

        Args:
            optimizer: Optimizer to schedule
            total_steps: Total number of training steps
            power: Power for polynomial decay
            min_lr_ratio: Minimum learning rate as ratio of base LR
            last_epoch: Last epoch index
        """
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super(PolynomialDecaySchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        if self.last_epoch >= self.total_steps:
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]

        decay_factor = (1 - float(self.last_epoch) / float(self.total_steps)) ** self.power
        return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * decay_factor)
                for base_lr in self.base_lrs]


class NoamSchedule(_LRScheduler):
    """
    Noam learning rate schedule from "Attention is All You Need".
    """

    def __init__(self,
                 optimizer: Optimizer,
                 model_size: int = 512,
                 warmup_steps: int = 4000,
                 last_epoch: int = -1):
        """
        Initialize Noam scheduler.

        Args:
            optimizer: Optimizer to schedule
            model_size: Model dimension (for scaling)
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch index
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super(NoamSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        step = max(1, self.last_epoch)
        scale_factor = (self.model_size ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale_factor for base_lr in self.base_lrs]


class CyclicLRSchedule(_LRScheduler):
    """
    Cyclic learning rate schedule.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 base_lr: float,
                 max_lr: float,
                 step_size: int,
                 mode: str = 'triangular',
                 gamma: float = 1.0,
                 last_epoch: int = -1):
        """
        Initialize cyclic LR scheduler.

        Args:
            optimizer: Optimizer to schedule
            base_lr: Lower boundary of learning rate
            max_lr: Upper boundary of learning rate
            step_size: Half period of the cycle
            mode: 'triangular', 'triangular2', or 'exp_range'
            gamma: Decay factor for 'exp_range' mode
            last_epoch: Last epoch index
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        # Override base_lrs to use our base_lr
        super(CyclicLRSchedule, self).__init__(optimizer, last_epoch)
        self.base_lrs = [base_lr] * len(optimizer.param_groups)

    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)

        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.last_epoch
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        return [lr] * len(self.base_lrs)


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 verbose: bool = False,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: Union[float, List[float]] = 0,
                 eps: float = 1e-8):
        """
        Initialize ReduceLROnPlateau scheduler.

        Args:
            optimizer: Optimizer to schedule
            mode: 'min' or 'max' - whether to minimize or maximize metric
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which LR is reduced
            verbose: Whether to print messages when LR is reduced
            threshold: Threshold for measuring improvement
            threshold_mode: 'rel' or 'abs' - relative or absolute threshold
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: Minimum learning rate(s)
            eps: Minimal decay applied to lr
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr if isinstance(min_lr, list) else [min_lr] * len(optimizer.param_groups)
        self.eps = eps

        # State variables
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.cooldown_counter = 0
        self.last_epoch = 0

        self._init_is_better()

    def _init_is_better(self):
        """Initialize comparison function based on mode."""
        if self.mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max'
            self.mode_worse = -float('inf')

    def is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return current < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return current > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs'
            return current > best + self.threshold

    def step(self, metric: float):
        """Step the scheduler with current metric value."""
        self.last_epoch += 1

        if self.best is None:
            self.best = metric
        elif self.is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    @property
    def in_cooldown(self) -> bool:
        """Check if scheduler is in cooldown period."""
        return self.cooldown_counter > 0

    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Reducing learning rate of group {i} to {new_lr:.4e}.')


# Scheduler registry
SCHEDULER_REGISTRY = {
    'linear_warmup': WarmupLinearSchedule,
    'cosine_warmup': WarmupCosineSchedule,
    'polynomial': PolynomialDecaySchedule,
    'noam': NoamSchedule,
    'cyclic': CyclicLRSchedule,
    'plateau': ReduceLROnPlateau
}


def get_scheduler(scheduler_name: str, optimizer: Optimizer, **kwargs) -> Union[_LRScheduler, ReduceLROnPlateau]:
    """
    Get learning rate scheduler by name.

    Args:
        scheduler_name: Name of the scheduler
        optimizer: Optimizer to schedule
        **kwargs: Additional arguments for scheduler

    Returns:
        Scheduler instance

    Raises:
        ValueError: If scheduler_name is not supported
    """
    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. "
                         f"Supported schedulers: {list(SCHEDULER_REGISTRY.keys())}")

    scheduler_class = SCHEDULER_REGISTRY[scheduler_name]
    return scheduler_class(optimizer, **kwargs)


def get_linear_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    last_epoch: int = -1) -> WarmupLinearSchedule:
    """
    Create linear schedule with warmup (commonly used in transformer training).

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch index

    Returns:
        Linear warmup scheduler
    """
    return WarmupLinearSchedule(
        optimizer=optimizer,
        warmup_steps=num_warmup_steps,
        total_steps=num_training_steps,
        last_epoch=last_epoch
    )


def get_cosine_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    min_lr_ratio: float = 0.0,
                                    last_epoch: int = -1) -> WarmupCosineSchedule:
    """
    Create cosine schedule with warmup.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of base LR
        last_epoch: Last epoch index

    Returns:
        Cosine warmup scheduler
    """
    return WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=num_warmup_steps,
        total_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
        last_epoch=last_epoch
    )