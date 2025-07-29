"""
Training classes for QDER models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Callable, Any, List
from tqdm import tqdm
import logging
import time
import os
from .training_utils import TrainingLogger, EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class QDERTrainer:
    """
    Trainer class for QDER models.

    Handles the complete training loop including validation, checkpointing,
    early stopping, and logging.
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 scheduler: Optional[_LRScheduler] = None,
                 device: str = 'cuda',
                 gradient_clip_norm: Optional[float] = 1.0,
                 accumulate_grad_batches: int = 1,
                 mixed_precision: bool = False) -> None:
        """
        Initialize the trainer.

        Args:
            model: Model to train
            optimizer: Optimizer for training
            criterion: Loss function
            scheduler: Optional learning rate scheduler
            device: Device to train on
            gradient_clip_norm: Gradient clipping norm (None to disable)
            accumulate_grad_batches: Number of batches to accumulate gradients
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.mixed_precision = mixed_precision

        # Move model to device
        self.model.to(device)
        self.criterion.to(device)

        # Initialize mixed precision if enabled
        self.scaler = None
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.training_history = []

        # Initialize logger
        self.logger = TrainingLogger()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        # Initialize gradient accumulation
        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} Training')

        for batch_idx, batch in enumerate(pbar):
            batch_loss = self._train_step(batch, batch_idx)

            total_loss += batch_loss
            num_batches += 1
            num_samples += batch['label'].size(0)

            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{self._get_current_lr():.2e}'})

        # Calculate epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            'train_samples': num_samples,
            'learning_rate': self._get_current_lr()
        }

        return epoch_metrics

    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        """
        Perform a single training step.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Batch loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass with optional mixed precision
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    query_token_type_ids=batch['query_token_type_ids'],
                    query_entity_emb=batch['query_entity_emb'],
                    doc_input_ids=batch['doc_input_ids'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    doc_token_type_ids=batch['doc_token_type_ids'],
                    doc_entity_emb=batch['doc_entity_emb'],
                    query_entity_mask=batch.get('query_entity_mask'),
                    doc_entity_mask=batch.get('doc_entity_mask'),
                    doc_scores=batch.get('doc_score')
                )

                loss = self.criterion(outputs['score'], batch['label'].float())

            # Scale loss for gradient accumulation
            loss = loss / self.accumulate_grad_batches

            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()

        else:
            # Standard precision forward pass
            outputs = self.model(
                query_input_ids=batch['query_input_ids'],
                query_attention_mask=batch['query_attention_mask'],
                query_token_type_ids=batch['query_token_type_ids'],
                query_entity_emb=batch['query_entity_emb'],
                doc_input_ids=batch['doc_input_ids'],
                doc_attention_mask=batch['doc_attention_mask'],
                doc_token_type_ids=batch['doc_token_type_ids'],
                doc_entity_emb=batch['doc_entity_emb'],
                query_entity_mask=batch.get('query_entity_mask'),
                doc_entity_mask=batch.get('doc_entity_mask'),
                doc_scores=batch.get('doc_score')
            )

            loss = self.criterion(outputs['score'], batch['label'].float())

            # Scale loss for gradient accumulation
            loss = loss / self.accumulate_grad_batches

            # Backward pass
            loss.backward()

        # Update parameters if we've accumulated enough gradients
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self._optimizer_step()
            self.global_step += 1

        return loss.item() * self.accumulate_grad_batches

    def _optimizer_step(self) -> None:
        """Perform optimizer step with optional gradient clipping."""
        if self.mixed_precision and self.scaler is not None:
            # Gradient clipping with mixed precision
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard gradient clipping and optimizer step
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()

        # Zero gradients
        self.optimizer.zero_grad()

        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()

    def validate(self, val_loader: DataLoader,
                 metric_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            metric_fn: Optional metric function for evaluation

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        predictions = []
        targets = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')

            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    query_token_type_ids=batch['query_token_type_ids'],
                    query_entity_emb=batch['query_entity_emb'],
                    doc_input_ids=batch['doc_input_ids'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    doc_token_type_ids=batch['doc_token_type_ids'],
                    doc_entity_emb=batch['doc_entity_emb'],
                    query_entity_mask=batch.get('query_entity_mask'),
                    doc_entity_mask=batch.get('doc_entity_mask'),
                    doc_scores=batch.get('doc_score')
                )

                loss = self.criterion(outputs['score'], batch['label'].float())

                total_loss += loss.item()
                num_batches += 1

                # Store predictions and targets for metrics
                predictions.extend(outputs['score'].cpu().numpy())
                targets.extend(batch['label'].cpu().numpy())

                # Update progress bar
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})

        # Calculate validation metrics
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_samples': len(predictions)
        }

        # Add custom metrics if provided
        if metric_fn is not None:
            custom_metrics = metric_fn(predictions, targets)
            val_metrics.update(custom_metrics)

        return val_metrics

    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 10,
            early_stopping: Optional[EarlyStopping] = None,
            model_checkpoint: Optional[ModelCheckpoint] = None,
            metric_fn: Optional[Callable] = None,
            log_every: int = 1) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            early_stopping: Optional early stopping callback
            model_checkpoint: Optional model checkpointing callback
            metric_fn: Optional metric function for evaluation
            log_every: Log every N epochs

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")

        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader)
            training_history['train_loss'].append(train_metrics['train_loss'])
            training_history['learning_rate'].append(train_metrics['learning_rate'])

            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader, metric_fn)
                training_history['val_loss'].append(val_metrics['val_loss'])

            # Calculate epoch time
            epoch_time = time.time() - start_time

            # Combine all metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch_time': epoch_time}

            # Log metrics
            if (epoch + 1) % log_every == 0:
                self._log_epoch_metrics(epoch + 1, epoch_metrics)

            # Early stopping check
            if early_stopping is not None and val_loader is not None:
                early_stopping(val_metrics.get('val_loss', float('inf')))
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                    break

            # Model checkpointing
            if model_checkpoint is not None:
                metric_value = val_metrics.get('val_loss', train_metrics['train_loss'])
                model_checkpoint(self.model, metric_value, epoch + 1)

            # Store in training history
            self.training_history.append(epoch_metrics)

        logger.info("Training completed")
        return training_history

    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for an epoch."""
        log_str = f"Epoch {epoch:3d} | "
        log_str += f"Train Loss: {metrics['train_loss']:.4f} | "

        if 'val_loss' in metrics:
            log_str += f"Val Loss: {metrics['val_loss']:.4f} | "

        log_str += f"LR: {metrics['learning_rate']:.2e} | "
        log_str += f"Time: {metrics['epoch_time']:.1f}s"

        logger.info(log_str)

    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, filepath: str, additional_info: Dict[str, Any] = None) -> None:
        """
        Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if additional_info:
            checkpoint.update(additional_info)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> None:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', None)
        self.training_history = checkpoint.get('training_history', [])

        logger.info(f"Checkpoint loaded from {filepath}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': self.device,
            'mixed_precision': self.mixed_precision
        }