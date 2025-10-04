#!/usr/bin/env python3
"""
Training script for QDER models.
Supports distributed training and various model configurations.
"""

import os
import time
import json
import torch
import collections
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# Import from src modules
from src.models.model_factory import create_model, validate_model_args
from src.data.dataset import QDERDataset
from src.data.dataloader import QDERDataLoader
from src.training.trainer import QDERTrainer
from src.training.loss_functions import get_loss_function
from src.training.training_utils import (
    EarlyStopping, ModelCheckpoint, TrainingLogger,
    set_seed, get_device, count_parameters
)
from src.evaluation.evaluator import QDERModelEvaluator
from src.evaluation.metrics import get_metric
from src.utils.io_utils import save_checkpoint, check_dir
from src.utils.common_utils import epoch_time
from src.utils.arg_parsers import create_training_parser, setup_args_and_logging


def load_tfidf_weights(path):
    """Load TF-IDF weights from file."""
    res = collections.defaultdict(dict)
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Loading TF-IDF weights"):
            item = json.loads(line)
            for entry in item['tokens']:
                res[item['doc_id']][entry['token']] = entry['tfidf']
    return res


def create_datasets(args, tokenizer):
    """Create training and validation datasets."""
    print('Loading TF-IDF weights...' if args.tfidf_weights else 'No TF-IDF weights provided')
    tfidf_weights = load_tfidf_weights(args.tfidf_weights) if args.tfidf_weights else None
    if tfidf_weights:
        print('[Done]')

    print('Creating training dataset...')
    train_dataset = QDERDataset(
        dataset=args.train_data,
        tokenizer=tokenizer,
        train=True,
        max_len=args.max_len
    )
    print(f'Training dataset created with {len(train_dataset)} examples')

    val_dataset = None
    if args.dev_data:
        print('Creating validation dataset...')
        val_dataset = QDERDataset(
            dataset=args.dev_data,
            tokenizer=tokenizer,
            train=False,
            max_len=args.max_len
        )
        print(f'Validation dataset created with {len(val_dataset)} examples')

    return train_dataset, val_dataset, tfidf_weights


def create_data_loaders(train_dataset, val_dataset, args):
    """Create data loaders for training and validation."""
    print('Creating data loaders...')
    print(f'Number of workers: {args.num_workers}')
    print(f'Batch size: {args.batch_size}')

    train_loader = QDERDataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if hasattr(args, 'pin_memory') else False
    )

    val_loader = None
    if val_dataset:
        eval_batch_size = getattr(args, 'eval_batch_size', args.batch_size)
        val_loader = QDERDataLoader(
            dataset=val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory if hasattr(args, 'pin_memory') else False
        )

    print('[Done]')
    return train_loader, val_loader


def setup_training_components(model, args, train_loader):
    """Setup optimizer, scheduler, and loss function."""
    # Loss function
    criterion = get_loss_function('bce')

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=getattr(args, 'weight_decay', 0.01)
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    return criterion, optimizer, scheduler


def setup_callbacks(args):
    """Setup training callbacks."""
    callbacks = {}

    # Early stopping
    if hasattr(args, 'patience') and args.patience > 0:
        callbacks['early_stopping'] = EarlyStopping(
            patience=args.patience,
            mode='min',  # Minimize validation loss
            verbose=True
        )

    # Model checkpointing
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    callbacks['checkpoint'] = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=True
    )

    # Training logger
    callbacks['logger'] = TrainingLogger(log_dir=args.output_dir)

    return callbacks


def train_model(model, train_loader, val_loader, args, device):
    """Main training function."""
    # Setup training components
    criterion, optimizer, scheduler = setup_training_components(model, args, train_loader)

    # Setup callbacks
    callbacks = setup_callbacks(args)

    # Create trainer
    trainer = QDERTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        gradient_clip_norm=getattr(args, 'gradient_clip', 1.0),
        mixed_precision=getattr(args, 'mixed_precision', False)
    )

    print(f"Starting training for {args.epochs} epochs")
    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")

    # Training loop
    best_metric = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')

    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training phase
        train_metrics = trainer.train_epoch(train_loader)

        # Validation phase
        val_metrics = {}
        if val_loader is not None and (epoch + 1) % args.eval_every == 0:
            evaluator = QDERModelEvaluator(model, device)

            # Get predictions for TREC evaluation
            results = evaluator.evaluate(val_loader)

            # Save temporary run file for evaluation
            temp_run_file = os.path.join(args.output_dir, f'temp_epoch_{epoch + 1}.run')
            from src.evaluation.ranking_utils import save_trec_run
            save_trec_run(temp_run_file, results)

            # Calculate metric
            if args.qrels:
                metric_value = get_metric(args.qrels, temp_run_file, args.metric)
                val_metrics[f'val_{args.metric}'] = metric_value

                # Save best model
                if metric_value < best_metric:  # Assuming lower is better for loss-like metrics
                    best_metric = metric_value
                    save_checkpoint(best_model_path, model, optimizer, scheduler,
                                    epoch + 1, val_metrics)
                    print(f"New best {args.metric}: {metric_value:.4f}")

                # Clean up temp file
                os.remove(temp_run_file)

            # Compute validation loss
            val_loss_metrics = trainer.validate(val_loader)
            val_metrics.update(val_loss_metrics)

        # Calculate epoch time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Log metrics
        all_metrics = {**train_metrics, **val_metrics}
        print(f'Epoch {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_metrics["train_loss"]:.4f}', end='')
        if val_metrics:
            print(f' | Val Loss: {val_metrics.get("val_loss", 0.0):.4f}', end='')
            if f'val_{args.metric}' in val_metrics:
                print(f' | Val {args.metric}: {val_metrics[f"val_{args.metric}"]:.4f}', end='')
        print()

        # Early stopping check
        if 'early_stopping' in callbacks and val_metrics:
            stop_metric = val_metrics.get('val_loss', float('inf'))
            if callbacks['early_stopping'](stop_metric, model):
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        # Save checkpoint every N epochs
        if hasattr(args, 'save_every') and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                            epoch + 1, all_metrics)

    print("\nTraining completed!")
    return best_metric


def main():
    # Parse arguments
    parser = create_training_parser()
    args = setup_args_and_logging(parser)

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = get_device(args.cuda)

    # Create output directory
    check_dir(args.output_dir)

    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    try:
        # Validate model arguments
        validate_model_args(args)

        # Create model
        print("Creating model...")
        model = create_model(args)
        print(f"Model created: {model.__class__.__name__}")
        print(f"Parameters: {count_parameters(model):,}")

        # Move model to device
        model.to(device)

        # Get tokenizer
        from transformers import AutoTokenizer
        from src.models.model_factory import get_pretrained_model_name
        pretrained_name = get_pretrained_model_name(args.text_enc)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

        # Create datasets
        train_dataset, val_dataset, tfidf_weights = create_datasets(args, tokenizer)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, args)

        # Resume from checkpoint if specified
        if hasattr(args, 'resume') and args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            from src.utils.io_utils import load_checkpoint
            load_checkpoint(args.resume, model, device=str(device))

        # Train model
        best_metric = train_model(model, train_loader, val_loader, args, device)

        # Save final model
        final_model_path = os.path.join(args.output_dir, 'final_model.pt')
        save_checkpoint(final_model_path, model)

        print(f"\nTraining completed successfully!")
        print(f"Best {args.metric}: {best_metric:.4f}")
        print(f"Models saved to: {args.output_dir}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()