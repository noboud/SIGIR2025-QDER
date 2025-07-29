"""
Data processing and loading modules for QDER.

This package contains:
- Dataset classes for loading and processing QDER data
- DataLoader wrappers
- Text preprocessing utilities
- Data loading and I/O utilities
- Entity linking functionality
"""

from .dataset import QDERDataset
from .dataloader import QDERDataLoader
from .preprocessing import preprocess_text, prepare_text_for_model
from .data_utils import (
    load_entity_embeddings,
    load_queries,
    load_qrels,
    load_run_file,
    load_docs_from_jsonl,
    save_jsonl,
    validate_dataset_file
)
from .entity_linking import WATEntityLinker

__all__ = [
    'QDERDataset',
    'QDERDataLoader',
    'preprocess_text',
    'prepare_text_for_model',
    'load_entity_embeddings',
    'load_queries',
    'load_qrels',
    'load_run_file',
    'load_docs_from_jsonl',
    'save_jsonl',
    'validate_dataset_file',
    'WATEntityLinker'
]