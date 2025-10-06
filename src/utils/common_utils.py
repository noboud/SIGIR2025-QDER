"""
Common utility functions for QDER project.

This module provides general utility functions that are used
across different components of the project.
"""

import time
import random
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """
    Calculate elapsed time in minutes and seconds.

    Args:
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        Tuple of (minutes, seconds)
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def get_device(cuda_device: Optional[int] = None,
               use_cuda: bool = True) -> torch.device:
    """
    Get appropriate device for computation.

    Args:
        cuda_device: Specific CUDA device number
        use_cuda: Whether to use CUDA if available

    Returns:
        PyTorch device object
    """
    if use_cuda and torch.cuda.is_available():
        if cuda_device is not None:
            device = torch.device(f'cuda:{cuda_device}')
        else:
            device = torch.device('cuda')
        logger.info(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        logger.info("Using device: CPU")

    return device


def print_model_summary(model: nn.Module,
                        input_size: Optional[Tuple] = None) -> None:
    """
    Print summary of model architecture and parameters.

    Args:
        model: PyTorch model
        input_size: Input tensor size for forward pass estimation
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    if input_size:
        # Estimate model size and forward/backward pass memory
        param_size = total_params * 4 / (1024 ** 2)  # Assume float32
        print(f"Estimated model size: {param_size:.2f} MB")

    print("=" * 70)


def calculate_metrics(predictions: List[float],
                      targets: List[float],
                      threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate basic classification metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Convert to binary predictions
    binary_preds = (predictions > threshold).astype(int)
    binary_targets = targets.astype(int)

    # Calculate metrics
    tp = np.sum((binary_preds == 1) & (binary_targets == 1))
    tn = np.sum((binary_preds == 0) & (binary_targets == 0))
    fp = np.sum((binary_preds == 1) & (binary_targets == 0))
    fn = np.sum((binary_preds == 0) & (binary_targets == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def tensor_to_list(tensor: torch.Tensor) -> List:
    """
    Convert tensor to list, handling different tensor types.

    Args:
        tensor: Input tensor

    Returns:
        List representation of tensor
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.tolist()


def list_to_tensor(data: List,
                   dtype: torch.dtype = torch.float32,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert list to tensor with specified type and device.

    Args:
        data: Input list
        dtype: Target tensor dtype
        device: Target device

    Returns:
        Tensor representation of list
    """
    tensor = torch.tensor(data, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def safe_divide(numerator: float,
                denominator: float,
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero

    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def moving_average(values: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average of values.

    Args:
        values: List of values
        window_size: Size of moving window

    Returns:
        List of moving averages
    """
    if len(values) < window_size:
        return values

    moving_avgs = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        moving_avgs.append(sum(window) / window_size)

    return moving_avgs


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with nested keys.

    Args:
        d: Flattened dictionary
        sep: Separator used in nested keys

    Returns:
        Unflattened nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def batch_iterator(data: List[Any], batch_size: int):
    """
    Create batches from a list of data.

    Args:
        data: List of data items
        batch_size: Size of each batch

    Yields:
        Batches of data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def normalize_scores(scores: List[float],
                     method: str = 'min_max') -> List[float]:
    """
    Normalize scores using different methods.

    Args:
        scores: List of scores to normalize
        method: Normalization method ('min_max', 'z_score', 'softmax')

    Returns:
        Normalized scores
    """
    scores = np.array(scores)

    if method == 'min_max':
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return [1.0] * len(scores)
        return ((scores - min_score) / (max_score - min_score)).tolist()

    elif method == 'z_score':
        mean_score = scores.mean()
        std_score = scores.std()
        if std_score == 0:
            return [0.0] * len(scores)
        return ((scores - mean_score) / std_score).tolist()

    elif method == 'softmax':
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        return (exp_scores / np.sum(exp_scores)).tolist()

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage statistics
    """
    memory_info = {}

    # PyTorch GPU memory
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

    # System memory
    try:
        import psutil
        process = psutil.Process()
        memory_info['cpu_memory'] = process.memory_info().rss / (1024 ** 3)  # GB
        memory_info['cpu_memory_percent'] = process.memory_percent()
    except ImportError:
        logger.warning("psutil not available, cannot get CPU memory info")

    return memory_info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


def get_model_device(model: nn.Module) -> torch.device:
    """
    Get the device that a model is on.

    Args:
        model: PyTorch model

    Returns:
        Device of the model
    """
    return next(model.parameters()).device


def move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Move object to specified device (recursively handles dicts and lists).

    Args:
        obj: Object to move (tensor, dict, list, etc.)
        device: Target device

    Returns:
        Object moved to device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(val, device) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj


def ensure_list(obj: Any) -> List[Any]:
    """
    Ensure object is a list.

    Args:
        obj: Object to convert to list

    Returns:
        List containing the object(s)
    """
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def filter_dict(d: Dict[str, Any],
                keys: Optional[List[str]] = None,
                exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Filter dictionary by including or excluding specific keys.

    Args:
        d: Dictionary to filter
        keys: Keys to include (if None, include all)
        exclude_keys: Keys to exclude

    Returns:
        Filtered dictionary
    """
    if keys is not None:
        result = {k: v for k, v in d.items() if k in keys}
    else:
        result = d.copy()

    if exclude_keys is not None:
        result = {k: v for k, v in result.items() if k not in exclude_keys}

    return result


def deep_update(base_dict: Dict[str, Any],
                update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep update of nested dictionary.

    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates

    Returns:
        Updated dictionary
    """
    result = base_dict.copy()

    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def get_timestamp() -> str:
    """
    Get current timestamp as string.

    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def create_experiment_dir(base_dir: Union[str, Path],
                          experiment_name: Optional[str] = None) -> Path:
    """
    Create directory for experiment with timestamp.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name

    Returns:
        Path to created experiment directory
    """
    base_path = Path(base_dir)

    if experiment_name:
        dir_name = f"{experiment_name}_{get_timestamp()}"
    else:
        dir_name = f"experiment_{get_timestamp()}"

    exp_dir = base_path / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def load_config_from_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load configuration from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Configuration dictionary if available
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Try different keys where config might be stored
        for key in ['config', 'model_config', 'args']:
            if key in checkpoint:
                return checkpoint[key]

        logger.warning(f"No configuration found in checkpoint {checkpoint_path}")
        return None

    except Exception as e:
        logger.error(f"Error loading config from checkpoint {checkpoint_path}: {e}")
        return None


def log_metrics(metrics: Dict[str, float],
                step: Optional[int] = None,
                prefix: str = "") -> None:
    """
    Log metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
        prefix: Optional prefix for metric names
    """
    if step is not None:
        logger.info(f"Step {step}:")

    for name, value in metrics.items():
        metric_name = f"{prefix}{name}" if prefix else name
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


def setup_logging(log_level: str = "INFO",
                  log_file: Optional[Union[str, Path]] = None) -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

    logger.info(f"Logging setup complete (level: {log_level})")
    if log_file:
        logger.info(f"Log file: {log_file}")