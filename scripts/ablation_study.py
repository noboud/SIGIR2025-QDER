#!/usr/bin/env python3
"""
Ablation Study Script for CADR Model Interactions

This script performs comprehensive ablation studies on different interaction patterns
in the CADR model to understand the contribution of each interaction type.
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.qder_ablation import QDERAblation
from src.data.dataset import QDERDataset
from src.data.dataloader import QDERDataLoader
from src.training.trainer import QDERTrainer
from src.evaluation.evaluator import QDERModelEvaluator
from src.evaluation.metrics import get_metric
from src.utils.io_utils import save_json, load_json
from src.evaluation.ranking_utils import save_trec_run
from src.utils.common_utils import setup_logging, get_device
from src.utils.arg_parsers import add_training_args, add_model_args


class AblationStudy:
    """
    Comprehensive ablation study for CADR model interaction patterns.

    This class manages the execution of ablation experiments across different
    interaction configurations to understand their individual contributions.
    """

    # Define interaction variants for ablation
    INTERACTION_VARIANTS = {
        'full': ['add', 'subtract', 'multiply'],
        'no_add': ['subtract', 'multiply'],
        'no_subtract': ['add', 'multiply'],
        'no_multiply': ['add', 'subtract'],
        'add_only': ['add'],
        'subtract_only': ['subtract'],
        'multiply_only': ['multiply'],
        'no_interactions': []
    }

    def __init__(self, config: Dict):
        """
        Initialize ablation study with configuration.

        Args:
            config: Configuration dictionary containing all experiment parameters
        """
        self.config = config
        self.device = get_device(config.get('device', 'cuda'))
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.output_dir / 'ablation_study.log'
        self.logger = setup_logging(log_file, level=logging.INFO)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])

        # Load datasets
        self._load_datasets()

        # Store results
        self.results = {}

    def _load_datasets(self):
        """Load training and validation datasets."""
        self.logger.info("Loading datasets...")

        # Training dataset
        self.train_dataset = QDERDataset(
            dataset=self.config['train_data'],
            tokenizer=self.tokenizer,
            train=True,
            max_len=self.config['max_length']
        )

        # Validation dataset
        self.val_dataset = QDERDataset(
            dataset=self.config['val_data'],
            tokenizer=self.tokenizer,
            train=False,
            max_len=self.config['max_length']
        )

        self.logger.info(f"Loaded {len(self.train_dataset)} training samples")
        self.logger.info(f"Loaded {len(self.val_dataset)} validation samples")

    def _create_data_loaders(self, variant_name: str):
        """Create data loaders for a specific variant."""
        train_loader = QDERDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0)
        )

        val_loader = QDERDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 0)
        )

        return train_loader, val_loader

    def _create_model(self, enabled_interactions: List[str]) -> QDERAblation:
        """
        Create a QDER model with specific interactions enabled.

        Args:
            enabled_interactions: List of interaction types to enable

        Returns:
            Configured QDERAblation model
        """
        model = QDERAblation(
            pretrained=self.config['pretrained_model'],
            use_scores=self.config.get('use_scores', True),
            use_entities=self.config.get('use_entities', True),
            score_method=self.config.get('score_method', 'bilinear'),
            enabled_interactions=enabled_interactions
        )

        return model.to(self.device)

    def _setup_training(self, model: nn.Module, train_loader) -> tuple:
        """
        Setup training components (optimizer, scheduler, criterion).

        Args:
            model: The model to train
            train_loader: Training data loader

        Returns:
            Tuple of (optimizer, scheduler, criterion)
        """
        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.get('epochs', 5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.get('warmup_steps', 1000),
            num_training_steps=total_steps
        )

        return optimizer, scheduler, criterion

    def _train_variant(self, variant_name: str, enabled_interactions: List[str]) -> Dict:
        """
        Train a specific variant and return results.

        Args:
            variant_name: Name of the variant being trained
            enabled_interactions: List of enabled interactions

        Returns:
            Dictionary containing training results and metrics
        """
        self.logger.info(f"Training variant: {variant_name}")
        self.logger.info(f"Enabled interactions: {enabled_interactions}")

        # Create variant-specific output directory
        variant_dir = self.output_dir / variant_name
        variant_dir.mkdir(exist_ok=True)

        # Create model and data loaders
        model = self._create_model(enabled_interactions)
        train_loader, val_loader = self._create_data_loaders(variant_name)

        # Setup training components
        optimizer, scheduler, criterion = self._setup_training(model, train_loader)

        # Create trainer
        trainer = QDERTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=self.device,
            gradient_clip_norm=self.config.get('gradient_clip_norm', 1.0)
        )

        # Train the model
        training_history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config.get('epochs', 5),
            log_every=self.config.get('eval_every', 1)
        )

        # Get best model path
        best_model_path = variant_dir / 'best_model.pt'

        # Save the final model state
        trainer.save_checkpoint(str(best_model_path))

        # Evaluate the model
        evaluator = QDERModelEvaluator(model, self.device)

        # Generate predictions and save run file
        run_file = variant_dir / f'{variant_name}_predictions.run'
        results_dict = evaluator.evaluate(val_loader)
        save_trec_run(str(run_file), results_dict)

        # Calculate evaluation metrics
        metrics = {}
        if self.config.get('qrels_path'):
            for metric_name in ['map', 'ndcg_cut_10', 'P_10', 'recall_1000']:
                try:
                    metric_value = get_metric(
                        qrels=self.config['qrels_path'],
                        run=str(run_file),
                        metric=metric_name
                    )
                    metrics[metric_name] = metric_value
                    self.logger.info(f"{variant_name} - {metric_name}: {metric_value:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate {metric_name}: {e}")

        # Compile results
        variant_results = {
            'variant_name': variant_name,
            'enabled_interactions': enabled_interactions,
            'metrics': metrics,
            'training_history': training_history,
            'model_path': str(best_model_path),
            'run_file': str(run_file)
        }

        # Save variant-specific results
        save_json(variant_results, str(variant_dir / 'results.json'))

        return variant_results

    def run_ablation_study(self) -> Dict:
        """
        Run the complete ablation study across all variants.

        Returns:
            Dictionary containing results for all variants
        """
        self.logger.info("Starting ablation study...")
        self.logger.info(f"Testing {len(self.INTERACTION_VARIANTS)} variants")

        # Run each variant
        for variant_name, enabled_interactions in self.INTERACTION_VARIANTS.items():
            try:
                variant_results = self._train_variant(variant_name, enabled_interactions)
                self.results[variant_name] = variant_results

            except Exception as e:
                self.logger.error(f"Error training variant {variant_name}: {e}")
                self.results[variant_name] = {
                    'variant_name': variant_name,
                    'enabled_interactions': enabled_interactions,
                    'error': str(e),
                    'metrics': {}
                }
                continue

        # Save complete results
        self._save_final_results()

        # Generate analysis and visualizations
        self._analyze_results()

        return self.results

    def _save_final_results(self):
        """Save final consolidated results."""
        self.logger.info("Saving final results...")

        # Save complete results
        results_file = self.output_dir / 'ablation_results.json'
        save_json(self.results, str(results_file))

        # Create summary table
        summary_data = []
        for variant_name, results in self.results.items():
            if 'error' in results:
                continue

            row = {
                'variant': variant_name,
                'interactions': ', '.join(results['enabled_interactions']) if results[
                    'enabled_interactions'] else 'none'
            }

            # Add metric columns
            for metric_name, value in results.get('metrics', {}).items():
                row[metric_name] = value

            summary_data.append(row)

        # Save as CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_file = self.output_dir / 'ablation_summary.csv'
            summary_df.to_csv(csv_file, index=False)
            self.logger.info(f"Summary saved to {csv_file}")

            # Save as LaTeX table
            latex_file = self.output_dir / 'ablation_summary.tex'
            with open(latex_file, 'w') as f:
                f.write(summary_df.to_latex(index=False, float_format='{:.4f}'.format))

    def _analyze_results(self):
        """Perform analysis on the ablation results."""
        self.logger.info("Analyzing results...")

        # Find best and worst performing variants for each metric
        analysis = {}

        metric_names = set()
        for results in self.results.values():
            if 'metrics' in results:
                metric_names.update(results['metrics'].keys())

        for metric in metric_names:
            metric_results = []
            for variant_name, results in self.results.items():
                if 'metrics' in results and metric in results['metrics']:
                    metric_results.append((variant_name, results['metrics'][metric]))

            if metric_results:
                metric_results.sort(key=lambda x: x[1], reverse=True)
                analysis[metric] = {
                    'best': metric_results[0],
                    'worst': metric_results[-1],
                    'all_results': metric_results
                }

        # Calculate improvement/degradation
        if 'full' in self.results and 'no_interactions' in self.results:
            full_metrics = self.results['full'].get('metrics', {})
            no_int_metrics = self.results['no_interactions'].get('metrics', {})

            improvements = {}
            for metric in metric_names:
                if metric in full_metrics and metric in no_int_metrics:
                    improvement = full_metrics[metric] - no_int_metrics[metric]
                    relative_improvement = improvement / no_int_metrics[metric] * 100
                    improvements[metric] = {
                        'absolute': improvement,
                        'relative_percent': relative_improvement
                    }

            analysis['interaction_contribution'] = improvements

        # Save analysis results
        analysis_file = self.output_dir / 'ablation_analysis.json'
        save_json(analysis, str(analysis_file))

        # Print key findings
        self._print_key_findings(analysis)

    def _print_key_findings(self, analysis: Dict):
        """Print key findings from the ablation study."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("KEY FINDINGS")
        self.logger.info("=" * 50)

        for metric, results in analysis.items():
            if metric == 'interaction_contribution':
                self.logger.info(f"\nInteraction Contribution Analysis:")
                for m, improvement in results.items():
                    self.logger.info(f"  {m}: {improvement['relative_percent']:+.2f}% improvement")
            else:
                best_variant, best_score = results['best']
                worst_variant, worst_score = results['worst']
                self.logger.info(f"\n{metric.upper()}:")
                self.logger.info(f"  Best: {best_variant} ({best_score:.4f})")
                self.logger.info(f"  Worst: {worst_variant} ({worst_score:.4f})")
                self.logger.info(f"  Difference: {best_score - worst_score:.4f}")


def create_config_from_args(args) -> Dict:
    """Create configuration dictionary from command line arguments."""
    config = {
        # Data paths
        'train_data': args.train_data,
        'val_data': args.val_data,
        'qrels_path': args.qrels,
        'tfidf_weights_path': args.tfidf_weights,

        # Model configuration
        'pretrained_model': args.pretrained_model,
        'use_scores': args.use_scores,
        'use_entities': args.use_entities,
        'score_method': args.score_method,
        'max_length': args.max_length,

        # Training configuration
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'eval_every': args.eval_every,

        # System configuration
        'device': args.device,
        'num_workers': args.num_workers,
        'output_dir': args.output_dir,
    }

    return config


def main():
    """Main function to run the ablation study."""
    parser = argparse.ArgumentParser(
        description="Run ablation study on CADR model interactions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--train-data', required=True,
                        help='Path to training data file')
    parser.add_argument('--val-data', required=True,
                        help='Path to validation data file')
    parser.add_argument('--qrels', required=True,
                        help='Path to qrels file for evaluation')
    parser.add_argument('--tfidf-weights', default=None,
                        help='Path to TF-IDF weights file')

    # Model arguments
    add_model_args(parser)
    parser.add_argument('--pretrained-model', default='bert-base-uncased',
                        help='Pretrained model name')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--use-scores', action='store_true',
                        help='Use retrieval scores in model')
    parser.add_argument('--use-entities', action='store_true',
                        help='Use entity embeddings in model')
    parser.add_argument('--score-method', choices=['linear', 'bilinear'],
                        default='bilinear', help='Scoring method')

    # Training arguments
    add_training_args(parser)
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Warmup steps for scheduler')
    parser.add_argument('--eval-every', type=int, default=1,
                        help='Evaluate every N epochs')

    # System arguments
    add_general_args(parser)
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for results')

    args = parser.parse_args()

    # Create configuration
    config = create_config_from_args(args)

    # Initialize and run ablation study
    study = AblationStudy(config)
    results = study.run_ablation_study()

    print(f"\nAblation study completed!")
    print(f"Results saved to: {config['output_dir']}")

    return results


if __name__ == '__main__':
    main()