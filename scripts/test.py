#!/usr/bin/env python3
"""
Evaluation script for QDER models.
Supports model evaluation and TREC run generation.
"""

import os
import torch
import argparse
from pathlib import Path

# Import from src modules
from src.models.model_factory import create_model_from_config, load_model_from_checkpoint
from src.data.dataset import QDERDataset
from src.data.dataloader import QDERDataLoader
from src.evaluation.evaluator import QDERModelEvaluator
from src.evaluation.metrics import get_metric, compute_ranking_metrics
from src.evaluation.ranking_utils import save_trec_run, validate_run_file
from src.utils.io_utils import load_checkpoint, ensure_dir_exists
from src.utils.common_utils import get_device, setup_logging, count_parameters
from src.utils.arg_parsers import create_evaluation_parser, setup_args_and_logging


def load_model_and_tokenizer(args, device):
    """Load model and tokenizer from checkpoint."""
    print("Loading model from checkpoint...")

    try:
        # Try to load using model factory (preferred method)
        model = load_model_from_checkpoint(args.checkpoint, device=device)
        print(f"Model loaded successfully: {model.__class__.__name__}")

    except Exception as e:
        print(f"Failed to load with model factory: {e}")
        print("Attempting manual loading...")

        # Fallback: create model from args and load state dict
        model = create_model(args)
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Try different keys for state dict
        state_dict_keys = ['model_state_dict', 'state_dict']
        loaded = False

        for key in state_dict_keys:
            if key in checkpoint:
                model.load_state_dict(checkpoint[key])
                loaded = True
                break

        if not loaded:
            # Assume entire checkpoint is state dict
            model.load_state_dict(checkpoint)

        model.to(device)
        print("Model loaded with manual method")

    # Get tokenizer
    from transformers import AutoTokenizer
    from src.models.model_factory import get_pretrained_model_name

    # Try to get pretrained name from model config or args
    if hasattr(model, 'pretrained'):
        pretrained_name = model.pretrained
    else:
        pretrained_name = get_pretrained_model_name(args.text_enc)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    print(f"Tokenizer loaded: {pretrained_name}")
    print(f"Model parameters: {count_parameters(model):,}")

    return model, tokenizer


def create_test_dataset(args, tokenizer):
    """Create test dataset."""
    print('Creating test dataset...')

    test_dataset = QDERDataset(
        dataset=args.test_data,
        tokenizer=tokenizer,
        train=False,
        max_len=args.max_len
    )

    print(f'Test dataset created with {len(test_dataset)} examples')
    return test_dataset


def create_test_loader(test_dataset, args):
    """Create test data loader."""
    eval_batch_size = getattr(args, 'eval_batch_size', args.batch_size)

    test_loader = QDERDataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=getattr(args, 'pin_memory', False)
    )

    print(f'Test loader created with batch size: {eval_batch_size}')
    return test_loader


def evaluate_model(model, test_loader, args, device):
    """Evaluate model and generate results."""
    print("Starting evaluation...")

    # Create evaluator
    evaluator = QDERModelEvaluator(model, device)

    # Get model predictions
    print("Getting model predictions...")
    results = evaluator.evaluate(test_loader)

    print(f"Evaluation completed for {len(results)} queries")

    # Save TREC run file if requested
    if args.save_run:
        print(f"Saving TREC run file to: {args.save_run}")
        ensure_dir_exists(args.save_run)
        save_trec_run(args.save_run, results, args.run_name)

        # Validate run file
        validation_results = validate_run_file(args.save_run, args.qrels if hasattr(args, 'qrels') else None)
        if validation_results['valid_format']:
            print("✓ TREC run file validation passed")
            if 'statistics' in validation_results:
                stats = validation_results['statistics']
                print(f"  - Total queries: {stats['total_queries']}")
                print(f"  - Total documents: {stats['total_documents']}")
                print(f"  - Avg docs per query: {stats['avg_docs_per_query']:.1f}")
        else:
            print("✗ TREC run file validation failed")
            for error in validation_results['errors']:
                print(f"  Error: {error}")

    return results


def compute_evaluation_metrics(run_file, args):
    """Compute evaluation metrics if qrels are provided."""
    if not hasattr(args, 'qrels') or not args.qrels:
        print("No qrels provided, skipping metric computation")
        return None

    print(f"Computing evaluation metrics using {args.qrels}")

    try:
        # Compute single metric
        metric_value = get_metric(args.qrels, run_file, args.metric)
        print(f"{args.metric.upper()}: {metric_value:.4f}")

        # Compute multiple metrics for comprehensive evaluation
        print("\nComprehensive evaluation:")
        all_metrics = compute_ranking_metrics(args.qrels, run_file)

        # Display key metrics
        key_metrics = ['map', 'ndcg', 'ndcg_cut_10', 'P_10', 'P_20', 'recip_rank']
        for metric_name in key_metrics:
            if metric_name in all_metrics:
                print(f"{metric_name.upper()}: {all_metrics[metric_name]:.4f}")

        return all_metrics

    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None


def save_evaluation_results(results, metrics, args):
    """Save evaluation results and metrics."""
    if not args.output_dir:
        return

    ensure_dir_exists(args.output_dir)

    # Save metrics
    if metrics:
        metrics_file = os.path.join(args.output_dir, 'evaluation_metrics.json')
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

    # Save evaluation summary
    summary_file = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("QDER Model Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model checkpoint: {args.checkpoint}\n")
        f.write(f"Test data: {args.test_data}\n")
        f.write(f"Total queries evaluated: {len(results)}\n")

        if metrics:
            f.write(f"\nEvaluation Metrics:\n")
            f.write("-" * 20 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

    print(f"Evaluation summary saved to: {summary_file}")


def run_additional_analysis(model, test_loader, args, device):
    """Run additional analysis if requested."""
    if not hasattr(args, 'save_embeddings') or not args.save_embeddings:
        return

    print("Extracting model embeddings...")
    evaluator = QDERModelEvaluator(model, device)

    embedding_data = evaluator.get_embeddings(test_loader)

    if embedding_data['embeddings'] is not None:
        # Save embeddings
        embeddings_file = os.path.join(args.output_dir, 'model_embeddings.pt')
        torch.save(embedding_data, embeddings_file)
        print(f"Model embeddings saved to: {embeddings_file}")


def main():
    # Parse arguments
    parser = create_evaluation_parser()
    args = setup_args_and_logging(parser)

    # Setup device
    device = get_device(args.cuda, args.use_cuda)

    # Create output directory if specified
    if hasattr(args, 'output_dir') and args.output_dir:
        from src.utils.io_utils import check_dir
        check_dir(args.output_dir)

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args, device)

        # Create test dataset
        test_dataset = create_test_dataset(args, tokenizer)

        # Create test loader
        test_loader = create_test_loader(test_dataset, args)

        # Evaluate model
        results = evaluate_model(model, test_loader, args, device)

        # Compute metrics if qrels provided
        metrics = None
        if args.save_run:
            metrics = compute_evaluation_metrics(args.save_run, args)

        # Run additional analysis
        if hasattr(args, 'output_dir') and args.output_dir:
            run_additional_analysis(model, test_loader, args, device)
            save_evaluation_results(results, metrics, args)

        print("\nEvaluation completed successfully!")

        # Print final summary
        print(f"\nSummary:")
        print(f"- Evaluated {len(results)} queries")
        if metrics and args.metric in metrics:
            print(f"- {args.metric.upper()}: {metrics[args.metric]:.4f}")
        if args.save_run:
            print(f"- TREC run saved to: {args.save_run}")

    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        raise


if __name__ == '__main__':
    main()