"""
Command-line argument parsers for QDER project.

This module provides reusable argument parser components for different
aspects of the QDER system (models, training, data, evaluation, analysis).
"""

import argparse
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add model-related arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    model_group = parser.add_argument_group('Model arguments')

    model_group.add_argument(
        '--text-enc',
        default='bert',
        choices=['bert', 'bert-large', 'distilbert', 'roberta', 'roberta-large',
                 'deberta', 'deberta-large', 'ernie', 'electra', 'electra-base',
                 'conv-bert', 't5', 't5-large'],
        help='Text encoder to use (default: bert)'
    )

    model_group.add_argument(
        '--use-scores',
        action='store_true',
        help='Whether to use document retrieval scores'
    )

    model_group.add_argument(
        '--use-entities',
        action='store_true',
        help='Whether to use entity embeddings'
    )

    model_group.add_argument(
        '--score-method',
        default='linear',
        choices=['linear', 'bilinear'],
        help='Scoring method (default: linear)'
    )

    model_group.add_argument(
        '--model-type',
        default='qder',
        choices=['qder', 'qder_ablation'],
        help='Type of model to use (default: qder)'
    )

    model_group.add_argument(
        '--enabled-interactions',
        nargs='+',
        default=['add', 'subtract', 'multiply'],
        choices=['add', 'subtract', 'multiply'],
        help='Enabled interaction types for ablation model'
    )

    return parser


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add training-related arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    training_group = parser.add_argument_group('Training arguments')

    training_group.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )

    training_group.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size (default: 8)'
    )

    training_group.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )

    training_group.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )

    training_group.add_argument(
        '--warmup-steps',
        type=int,
        default=1000,
        help='Number of warmup steps (default: 1000)'
    )

    training_group.add_argument(
        '--eval-every',
        type=int,
        default=1,
        help='Evaluate every N epochs (default: 1)'
    )

    training_group.add_argument(
        '--save-every',
        type=int,
        default=1,
        help='Save checkpoint every N epochs (default: 1)'
    )

    training_group.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        help='Gradient clipping norm (default: 1.0)'
    )

    training_group.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
    )

    training_group.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )

    return parser


def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add data-related arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    data_group = parser.add_argument_group('Data arguments')

    data_group.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training data'
    )

    data_group.add_argument(
        '--dev-data',
        type=str,
        help='Path to development data'
    )

    data_group.add_argument(
        '--test-data',
        type=str,
        help='Path to test data'
    )

    data_group.add_argument(
        '--max-len',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )

    data_group.add_argument(
        '--tfidf-weights',
        type=str,
        help='Path to TF-IDF weights file'
    )

    data_group.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of data loader workers (default: 0)'
    )

    data_group.add_argument(
        '--pin-memory',
        action='store_true',
        help='Pin memory for data loaders'
    )

    return parser


def add_evaluation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add evaluation-related arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    eval_group = parser.add_argument_group('Evaluation arguments')

    eval_group.add_argument(
        '--qrels',
        type=str,
        required=True,
        help='Path to qrels file'
    )

    eval_group.add_argument(
        '--metric',
        type=str,
        default='map',
        choices=['map', 'ndcg', 'ndcg_cut_10', 'ndcg_cut_20', 'P_10', 'P_20', 'recall_1000'],
        help='Evaluation metric (default: map)'
    )

    eval_group.add_argument(
        '--run-name',
        type=str,
        default='QDER',
        help='Name for TREC run files (default: QDER)'
    )

    eval_group.add_argument(
        '--save-run',
        type=str,
        help='Path to save TREC run file'
    )

    eval_group.add_argument(
        '--eval-batch-size',
        type=int,
        help='Batch size for evaluation (defaults to training batch size)'
    )

    return parser


def add_analysis_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add analysis-related arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    analysis_group = parser.add_argument_group('Analysis arguments')

    analysis_group.add_argument(
        '--analysis-type',
        type=str,
        choices=['attention', 'interaction', 'noise', 'embedding', 'ranking'],
        help='Type of analysis to perform'
    )

    analysis_group.add_argument(
        '--max-docs',
        type=int,
        help='Maximum number of documents to analyze'
    )

    analysis_group.add_argument(
        '--noise-levels',
        nargs='+',
        type=float,
        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        help='Noise levels for robustness analysis'
    )

    analysis_group.add_argument(
        '--n-runs',
        type=int,
        default=5,
        help='Number of runs for noise analysis (default: 5)'
    )

    analysis_group.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for query pairing (default: 0.7)'
    )

    return parser


def add_general_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add general arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    general_group = parser.add_argument_group('General arguments')

    general_group.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )

    general_group.add_argument(
        '--exp-name',
        type=str,
        help='Experiment name'
    )

    general_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    general_group.add_argument(
        '--cuda',
        type=int,
        default=0,
        help='CUDA device number (default: 0)'
    )

    general_group.add_argument(
        '--use-cuda',
        action='store_true',
        help='Use CUDA if available'
    )

    general_group.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    general_group.add_argument(
        '--log-file',
        type=str,
        help='Path to log file'
    )

    general_group.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    general_group.add_argument(
        '--save-config',
        action='store_true',
        help='Save configuration to output directory'
    )

    return parser


def create_base_parser(description: str = "QDER Model") -> argparse.ArgumentParser:
    """
    Create base argument parser with common arguments.

    Args:
        description: Description for the parser

    Returns:
        Base ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    parser = add_general_args(parser)
    return parser


def create_training_parser() -> argparse.ArgumentParser:
    """Create parser for training scripts."""
    parser = create_base_parser("Train QDER Model")
    parser = add_model_args(parser)
    parser = add_training_args(parser)
    parser = add_data_args(parser)
    parser = add_evaluation_args(parser)
    return parser


def create_evaluation_parser() -> argparse.ArgumentParser:
    """Create parser for evaluation scripts."""
    parser = create_base_parser("Evaluate QDER Model")
    parser = add_model_args(parser)
    parser = add_data_args(parser)
    parser = add_evaluation_args(parser)

    # Add model loading arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    return parser


def create_analysis_parser() -> argparse.ArgumentParser:
    """Create parser for analysis scripts."""
    parser = create_base_parser("Analyze QDER Model")
    parser = add_model_args(parser)
    parser = add_data_args(parser)
    parser = add_analysis_args(parser)

    # Add model loading arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    return parser


def validate_args(args) -> None:
    """
    Validate command-line arguments.

    Args:
        args: Parsed arguments namespace

    Raises:
        ValueError: If arguments are invalid
    """
    # Validate paths exist
    required_files = []

    if hasattr(args, 'train_data') and args.train_data:
        required_files.append(args.train_data)
    if hasattr(args, 'dev_data') and args.dev_data:
        required_files.append(args.dev_data)
    if hasattr(args, 'test_data') and args.test_data:
        required_files.append(args.test_data)
    if hasattr(args, 'qrels') and args.qrels:
        required_files.append(args.qrels)
    if hasattr(args, 'checkpoint') and args.checkpoint:
        required_files.append(args.checkpoint)
    if hasattr(args, 'config') and args.config:
        required_files.append(args.config)

    for file_path in required_files:
        if not Path(file_path).exists():
            raise ValueError(f"Required file not found: {file_path}")

    # Validate numeric arguments
    if hasattr(args, 'epochs') and args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if hasattr(args, 'batch_size') and args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if hasattr(args, 'learning_rate') and args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if hasattr(args, 'max_len') and args.max_len <= 0:
        raise ValueError("max_len must be positive")

    # Validate interactions for ablation
    if hasattr(args, 'model_type') and args.model_type == 'qder_ablation':
        if hasattr(args, 'enabled_interactions') and not args.enabled_interactions:
            raise ValueError("enabled_interactions cannot be empty for ablation model")

    logger.info("Argument validation passed")


def save_args(args, save_path: str) -> None:
    """
    Save arguments to JSON file.

    Args:
        args: Arguments namespace
        save_path: Path to save arguments
    """
    args_dict = vars(args)

    # Convert Path objects to strings
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)

    with open(save_path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    logger.info(f"Arguments saved to {save_path}")


def load_args(load_path: str) -> Dict[str, Any]:
    """
    Load arguments from JSON file.

    Args:
        load_path: Path to load arguments from

    Returns:
        Dictionary of arguments
    """
    with open(load_path, 'r') as f:
        args_dict = json.load(f)

    logger.info(f"Arguments loaded from {load_path}")
    return args_dict


def merge_args_with_config(args, config_path: str) -> argparse.Namespace:
    """
    Merge command-line arguments with configuration file.

    Args:
        args: Parsed command-line arguments
        config_path: Path to configuration file

    Returns:
        Merged arguments namespace
    """
    # Load configuration
    config = load_args(config_path)

    # Convert args to dict
    args_dict = vars(args)

    # Merge: command-line args override config
    for key, value in config.items():
        if key not in args_dict or args_dict[key] is None:
            args_dict[key] = value

    # Convert back to namespace
    merged_args = argparse.Namespace(**args_dict)

    logger.info(f"Arguments merged with config from {config_path}")
    return merged_args


def print_args(args) -> None:
    """
    Print arguments in a formatted way.

    Args:
        args: Arguments namespace
    """
    print("=" * 50)
    print("Configuration:")
    print("=" * 50)

    args_dict = vars(args)
    max_key_length = max(len(key) for key in args_dict.keys())

    for key, value in sorted(args_dict.items()):
        print(f"{key:<{max_key_length}} : {value}")

    print("=" * 50)


def create_experiment_args(base_args, experiment_name: str) -> argparse.Namespace:
    """
    Create experiment-specific arguments with modified output directory.

    Args:
        base_args: Base arguments
        experiment_name: Name of the experiment

    Returns:
        Modified arguments namespace
    """
    # Create copy of args
    exp_args = argparse.Namespace(**vars(base_args))

    # Modify output directory to include experiment name
    base_output = Path(exp_args.output_dir)
    exp_args.output_dir = str(base_output / experiment_name)

    # Set experiment name if not already set
    if not hasattr(exp_args, 'exp_name') or not exp_args.exp_name:
        exp_args.exp_name = experiment_name

    return exp_args


# Common argument combinations for different script types
def get_common_training_args():
    """Get commonly used training argument combinations."""
    return {
        'quick_test': {
            'epochs': 2,
            'batch_size': 4,
            'eval_every': 1,
            'save_every': 1
        },
        'full_training': {
            'epochs': 20,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'eval_every': 1,
            'patience': 5
        },
        'large_model': {
            'batch_size': 4,
            'learning_rate': 1e-5,
            'gradient_clip': 0.5,
            'warmup_steps': 2000
        }
    }


def get_common_analysis_args():
    """Get commonly used analysis argument combinations."""
    return {
        'quick_analysis': {
            'max_docs': 100,
            'n_runs': 3,
            'noise_levels': [0.01, 0.05, 0.1]
        },
        'full_analysis': {
            'max_docs': 1000,
            'n_runs': 10,
            'noise_levels': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        },
        'deep_analysis': {
            'max_docs': None,  # All documents
            'n_runs': 20,
            'noise_levels': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        }
    }


def apply_preset_args(args, preset_name: str, preset_type: str = 'training') -> argparse.Namespace:
    """
    Apply preset argument combinations.

    Args:
        args: Base arguments namespace
        preset_name: Name of preset to apply
        preset_type: Type of preset ('training' or 'analysis')

    Returns:
        Arguments with preset applied
    """
    if preset_type == 'training':
        presets = get_common_training_args()
    elif preset_type == 'analysis':
        presets = get_common_analysis_args()
    else:
        raise ValueError(f"Unknown preset type: {preset_type}")

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset_args = presets[preset_name]
    args_dict = vars(args)

    # Apply preset values
    for key, value in preset_args.items():
        if hasattr(args, key):
            setattr(args, key, value)

    logger.info(f"Applied {preset_type} preset: {preset_name}")
    return args


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add distributed training arguments.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    dist_group = parser.add_argument_group('Distributed training arguments')

    dist_group.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed training'
    )

    dist_group.add_argument(
        '--local-rank',
        type=int,
        default=0,
        help='Local rank for distributed training'
    )

    dist_group.add_argument(
        '--world-size',
        type=int,
        default=1,
        help='Number of processes for distributed training'
    )

    dist_group.add_argument(
        '--dist-backend',
        type=str,
        default='nccl',
        choices=['nccl', 'gloo', 'mpi'],
        help='Distributed backend (default: nccl)'
    )

    dist_group.add_argument(
        '--dist-url',
        type=str,
        default='env://',
        help='URL for distributed training initialization'
    )

    return parser


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add experiment management arguments.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    exp_group = parser.add_argument_group('Experiment management arguments')

    exp_group.add_argument(
        '--wandb',
        action='store_true',
        help='Use Weights & Biases logging'
    )

    exp_group.add_argument(
        '--wandb-project',
        type=str,
        default='qder',
        help='W&B project name (default: qder)'
    )

    exp_group.add_argument(
        '--wandb-entity',
        type=str,
        help='W&B entity name'
    )

    exp_group.add_argument(
        '--tensorboard',
        action='store_true',
        help='Use TensorBoard logging'
    )

    exp_group.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save model predictions'
    )

    exp_group.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Save model embeddings'
    )

    return parser


def create_complete_parser(script_type: str) -> argparse.ArgumentParser:
    """
    Create complete parser for different script types.

    Args:
        script_type: Type of script ('train', 'eval', 'analysis', 'data_prep')

    Returns:
        Complete ArgumentParser for the script type
    """
    if script_type == 'train':
        parser = create_training_parser()
        parser = add_distributed_args(parser)
        parser = add_experiment_args(parser)

    elif script_type == 'eval':
        parser = create_evaluation_parser()
        parser = add_experiment_args(parser)

    elif script_type == 'analysis':
        parser = create_analysis_parser()

    elif script_type == 'data_prep':
        parser = create_base_parser("Data Preparation for QDER")
        parser = add_data_args(parser)

        # Add data preparation specific args
        prep_group = parser.add_argument_group('Data preparation arguments')
        prep_group.add_argument('--input-file', type=str, required=True, help='Input data file')
        prep_group.add_argument('--output-file', type=str, required=True, help='Output data file')
        prep_group.add_argument('--format', choices=['jsonl', 'tsv', 'csv'], default='jsonl', help='Output format')
        prep_group.add_argument('--balance', action='store_true', help='Balance positive/negative examples')
        prep_group.add_argument('--max-examples', type=int, help='Maximum number of examples to process')

    else:
        raise ValueError(f"Unknown script type: {script_type}")

    return parser


def setup_args_and_logging(parser: argparse.ArgumentParser,
                           config_key: Optional[str] = None) -> argparse.Namespace:
    """
    Parse arguments, setup logging, and validate.

    Args:
        parser: ArgumentParser instance
        config_key: Optional key for loading specific config section

    Returns:
        Validated arguments namespace
    """
    # Parse arguments
    args = parser.parse_args()

    # Load config if specified
    if hasattr(args, 'config') and args.config:
        args = merge_args_with_config(args, args.config)

    # Setup logging
    log_level = getattr(args, 'log_level', 'INFO')
    log_file = getattr(args, 'log_file', None)

    from .common_utils import setup_logging
    setup_logging(log_level, log_file)

    # Validate arguments
    validate_args(args)

    # Save arguments if requested
    if hasattr(args, 'save_config') and args.save_config:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_args(args, output_dir / 'config.json')

    # Print arguments
    print_args(args)

    return args


def add_hyperparameter_search_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add hyperparameter search arguments.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Parser with added arguments
    """
    hp_group = parser.add_argument_group('Hyperparameter search arguments')

    hp_group.add_argument(
        '--hp-search',
        action='store_true',
        help='Enable hyperparameter search'
    )

    hp_group.add_argument(
        '--hp-search-space',
        type=str,
        help='Path to hyperparameter search space definition'
    )

    hp_group.add_argument(
        '--hp-search-trials',
        type=int,
        default=10,
        help='Number of hyperparameter search trials (default: 10)'
    )

    hp_group.add_argument(
        '--hp-search-method',
        type=str,
        default='random',
        choices=['random', 'grid', 'bayesian'],
        help='Hyperparameter search method (default: random)'
    )

    return parser


def create_grid_search_configs(base_args, search_space: Dict[str, List]) -> List[argparse.Namespace]:
    """
    Create configurations for grid search.

    Args:
        base_args: Base arguments
        search_space: Dictionary defining search space

    Returns:
        List of argument configurations
    """
    import itertools

    # Get all parameter combinations
    keys = search_space.keys()
    values = search_space.values()

    configs = []
    for combination in itertools.product(*values):
        # Create new config
        config_dict = vars(base_args).copy()

        # Update with current combination
        for key, value in zip(keys, combination):
            config_dict[key] = value

        configs.append(argparse.Namespace(**config_dict))

    return configs


def validate_hyperparameter_space(search_space: Dict[str, List]) -> None:
    """
    Validate hyperparameter search space.

    Args:
        search_space: Dictionary defining search space

    Raises:
        ValueError: If search space is invalid
    """
    if not search_space:
        raise ValueError("Search space cannot be empty")

    for param, values in search_space.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Parameter {param} must have a non-empty list of values")

    logger.info(f"Validated hyperparameter search space with {len(search_space)} parameters")


# Legacy function for backward compatibility
def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Legacy function - use add_general_args instead."""
    import warnings
    warnings.warn("add_common_args is deprecated, use add_general_args instead", DeprecationWarning)
    return add_general_args(parser)