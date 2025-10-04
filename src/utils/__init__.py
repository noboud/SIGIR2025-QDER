"""
Utility functions for QDER project.

This package contains:
- I/O utilities for reading/writing files and checkpoints
- Common utility functions for data processing
- Command-line argument parsers
- Helper functions for various tasks
"""

from .io_utils import (
    check_dir,
    save_trec_run,
    load_trec_run,
    save_features,
    save_checkpoint,
    load_checkpoint,
    write_to_file,
    read_jsonl,
    read_tsv_to_dict,
    ensure_dir_exists
)

from .common_utils import (
    count_parameters,
    count_trainable_parameters,
    epoch_time,
    set_random_seed,
    get_device,
    format_time,
    calculate_metrics,
    print_model_summary,
    tensor_to_list,
    list_to_tensor,
    safe_divide
)

from .arg_parsers import (
    add_model_args,
    add_training_args,
    add_data_args,
    add_evaluation_args,
    add_analysis_args,
    create_base_parser,
    validate_args,
    save_args,
    load_args
)

__all__ = [
    # I/O utilities
    'check_dir',
    'save_trec_run',
    'load_trec_run',
    'save_features',
    'save_checkpoint',
    'load_checkpoint',
    'write_to_file',
    'read_jsonl',
    'read_tsv_to_dict',
    'ensure_dir_exists',

    # Common utilities
    'count_parameters',
    'count_trainable_parameters',
    'epoch_time',
    'set_random_seed',
    'get_device',
    'format_time',
    'calculate_metrics',
    'print_model_summary',
    'tensor_to_list',
    'list_to_tensor',
    'safe_divide',

    # Argument parsers
    'add_model_args',
    'add_training_args',
    'add_data_args',
    'add_evaluation_args',
    'add_analysis_args',
    'create_base_parser',
    'validate_args',
    'save_args',
    'load_args',
]

__version__ = '1.0.0'