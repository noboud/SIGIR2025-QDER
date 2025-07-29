"""
I/O utilities for QDER project.

This module provides utilities for file I/O operations, including
saving/loading model checkpoints, TREC format files, and other
common file operations.
"""

import os
import json
import torch
import logging
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
import pickle
import gzip
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IOError(Exception):
    """Custom exception for I/O operations."""
    pass


def check_dir(path: Union[str, Path]) -> str:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to check/create

    Returns:
        String path to the directory
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        os.makedirs(path_str)
        logger.info(f"Created directory: {path_str}")
    return path_str


def ensure_dir_exists(file_path: Union[str, Path]) -> None:
    """
    Ensure the directory for a file path exists.

    Args:
        file_path: Path to file (directory will be created)
    """
    directory = os.path.dirname(str(file_path))
    if directory:
        check_dir(directory)


def save_trec_run(run_file: Union[str, Path],
                  results_dict: Dict[str, Dict[str, List[float]]],
                  run_name: str = "QDER") -> None:
    """
    Save results in TREC format.

    Args:
        run_file: Path to output run file
        results_dict: Results dictionary {query_id: {doc_id: [score, label]}}
        run_name: Name to use in TREC format

    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir_exists(run_file)

        with open(run_file, 'w') as writer:
            for query_id, doc_scores in results_dict.items():
                # Sort documents by score (descending)
                ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)

                for rank, (doc_id, score_info) in enumerate(ranked_docs, 1):
                    score = score_info[0] if isinstance(score_info, list) else score_info
                    # TREC format: query_id Q0 doc_id rank score run_name
                    writer.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

        logger.info(f"TREC run saved to {run_file}")

    except Exception as e:
        raise IOError(f"Error saving TREC run to {run_file}: {e}")


def load_trec_run(run_file: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load TREC format run file.

    Args:
        run_file: Path to TREC run file

    Returns:
        Dictionary {query_id: {doc_id: score}}

    Raises:
        IOError: If file cannot be read
    """
    try:
        run_data = {}

        with open(run_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        query_id = parts[0]
                        doc_id = parts[2]
                        score = float(parts[4])

                        if query_id not in run_data:
                            run_data[query_id] = {}
                        run_data[query_id][doc_id] = score
                    else:
                        logger.warning(f"Skipping malformed line {line_num} in {run_file}")

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line {line_num} in {run_file}: {e}")
                    continue

        logger.info(f"Loaded TREC run from {run_file} with {len(run_data)} queries")
        return run_data

    except FileNotFoundError:
        raise IOError(f"TREC run file not found: {run_file}")
    except Exception as e:
        raise IOError(f"Error loading TREC run from {run_file}: {e}")


def save_features(features_file: Union[str, Path], features: List[str]) -> None:
    """
    Save features to file (one per line).

    Args:
        features_file: Path to features file
        features: List of feature strings

    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir_exists(features_file)

        with open(features_file, 'w') as writer:
            for feature in features:
                writer.write(f"{feature}\n")

        logger.info(f"Saved {len(features)} features to {features_file}")

    except Exception as e:
        raise IOError(f"Error saving features to {features_file}: {e}")


def save_checkpoint(save_path: Union[str, Path],
                    model,
                    optimizer=None,
                    scheduler=None,
                    epoch: Optional[int] = None,
                    metrics: Optional[Dict[str, float]] = None,
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint with training state.

    Args:
        save_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: Scheduler state (optional)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
        additional_info: Additional information to save (optional)

    Raises:
        IOError: If checkpoint cannot be saved
    """
    try:
        ensure_dir_exists(save_path)

        checkpoint = {
            'model_state_dict': model.state_dict(),
        }

        # Add model configuration if available
        if hasattr(model, 'config'):
            checkpoint['model_config'] = model.config

        # Add training state
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics

        # Add additional information
        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    except Exception as e:
        raise IOError(f"Error saving checkpoint to {save_path}: {e}")


def load_checkpoint(load_path: Union[str, Path],
                    model,
                    optimizer=None,
                    scheduler=None,
                    device: str = 'cpu',
                    strict: bool = True) -> Dict[str, Any]:
    """
    Load model checkpoint and training state.

    Args:
        load_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary containing loaded information (epoch, metrics, etc.)

    Raises:
        IOError: If checkpoint cannot be loaded
    """
    try:
        checkpoint = torch.load(load_path, map_location=device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            # Assume entire checkpoint is model state dict
            model.load_state_dict(checkpoint, strict=strict)

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Return additional information
        info = {}
        for key in ['epoch', 'metrics', 'model_config']:
            if key in checkpoint:
                info[key] = checkpoint[key]

        logger.info(f"Checkpoint loaded from {load_path}")
        return info

    except FileNotFoundError:
        raise IOError(f"Checkpoint file not found: {load_path}")
    except Exception as e:
        raise IOError(f"Error loading checkpoint from {load_path}: {e}")


def write_to_file(data: List[str],
                  output_file: Union[str, Path],
                  mode: str = 'w',
                  encoding: str = 'utf-8') -> None:
    """
    Write list of strings to file.

    Args:
        data: List of strings to write
        output_file: Path to output file
        mode: File open mode ('w', 'a', etc.)
        encoding: File encoding

    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir_exists(output_file)

        with open(output_file, mode, encoding=encoding) as f:
            for line in data:
                f.write(f"{line}\n")

        logger.info(f"Wrote {len(data)} lines to {output_file}")

    except Exception as e:
        raise IOError(f"Error writing to {output_file}: {e}")


def read_jsonl(file_path: Union[str, Path],
               max_lines: Optional[int] = None,
               show_progress: bool = False) -> List[Dict[str, Any]]:
    """
    Read JSONL (JSON Lines) file.

    Args:
        file_path: Path to JSONL file
        max_lines: Maximum number of lines to read
        show_progress: Whether to show progress bar

    Returns:
        List of JSON objects

    Raises:
        IOError: If file cannot be read
    """
    try:
        data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f if not show_progress else tqdm(f, desc=f"Reading {Path(file_path).name}")

            for line_num, line in enumerate(lines, 1):
                if max_lines and line_num > max_lines:
                    break

                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue

        logger.info(f"Read {len(data)} objects from {file_path}")
        return data

    except FileNotFoundError:
        raise IOError(f"JSONL file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error reading JSONL file {file_path}: {e}")


def write_jsonl(data: List[Dict[str, Any]],
                output_file: Union[str, Path],
                mode: str = 'w',
                encoding: str = 'utf-8') -> None:
    """
    Write data to JSONL (JSON Lines) file.

    Args:
        data: List of dictionaries to write
        output_file: Path to output file
        mode: File open mode
        encoding: File encoding

    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir_exists(output_file)

        with open(output_file, mode, encoding=encoding) as f:
            for obj in data:
                json.dump(obj, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Wrote {len(data)} objects to {output_file}")

    except Exception as e:
        raise IOError(f"Error writing JSONL to {output_file}: {e}")


def read_tsv_to_dict(file_path: Union[str, Path],
                     key_col: int = 0,
                     value_col: int = 1,
                     delimiter: str = '\t',
                     has_header: bool = False) -> Dict[str, str]:
    """
    Read TSV file into dictionary.

    Args:
        file_path: Path to TSV file
        key_col: Column index for keys
        value_col: Column index for values
        delimiter: Field delimiter
        has_header: Whether file has header row

    Returns:
        Dictionary mapping keys to values

    Raises:
        IOError: If file cannot be read
    """
    try:
        result = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Skip header if present
                if has_header and line_num == 1:
                    continue

                try:
                    parts = line.strip().split(delimiter)
                    if len(parts) > max(key_col, value_col):
                        key = parts[key_col].strip()
                        value = parts[value_col].strip()
                        result[key] = value
                    else:
                        logger.warning(f"Skipping line {line_num}: insufficient columns")
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

        logger.info(f"Read {len(result)} entries from {file_path}")
        return result

    except FileNotFoundError:
        raise IOError(f"TSV file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error reading TSV file {file_path}: {e}")


def save_json(data: Any,
              output_file: Union[str, Path],
              indent: int = 2,
              ensure_ascii: bool = False) -> None:
    """
    Save data as JSON file.

    Args:
        data: Data to save
        output_file: Path to output file
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII encoding

    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir_exists(output_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.info(f"Saved JSON to {output_file}")

    except Exception as e:
        raise IOError(f"Error saving JSON to {output_file}: {e}")


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        IOError: If file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded JSON from {file_path}")
        return data

    except FileNotFoundError:
        raise IOError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise IOError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise IOError(f"Error loading JSON from {file_path}: {e}")


def save_pickle(data: Any,
                output_file: Union[str, Path],
                compress: bool = False) -> None:
    """
    Save data using pickle.

    Args:
        data: Data to save
        output_file: Path to output file
        compress: Whether to compress with gzip

    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir_exists(output_file)

        if compress:
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)

        logger.info(f"Saved pickle to {output_file}")

    except Exception as e:
        raise IOError(f"Error saving pickle to {output_file}: {e}")


def load_pickle(file_path: Union[str, Path],
                compressed: bool = False) -> Any:
    """
    Load data from pickle file.

    Args:
        file_path: Path to pickle file
        compressed: Whether file is gzip compressed

    Returns:
        Loaded data

    Raises:
        IOError: If file cannot be read
    """
    try:
        if compressed:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

        logger.info(f"Loaded pickle from {file_path}")
        return data

    except FileNotFoundError:
        raise IOError(f"Pickle file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error loading pickle from {file_path}: {e}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Raises:
        IOError: If file cannot be accessed
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        raise IOError(f"Error getting size of {file_path}: {e}")


def count_lines(file_path: Union[str, Path]) -> int:
    """
    Count number of lines in a file efficiently.

    Args:
        file_path: Path to file

    Returns:
        Number of lines

    Raises:
        IOError: If file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        raise IOError(f"Error counting lines in {file_path}: {e}")


def backup_file(file_path: Union[str, Path],
                backup_suffix: str = '.bak') -> str:
    """
    Create a backup copy of a file.

    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix for backup file

    Returns:
        Path to backup file

    Raises:
        IOError: If backup cannot be created
    """
    try:
        backup_path = str(file_path) + backup_suffix

        import shutil
        shutil.copy2(file_path, backup_path)

        logger.info(f"Created backup: {backup_path}")
        return backup_path

    except Exception as e:
        raise IOError(f"Error creating backup of {file_path}: {e}")