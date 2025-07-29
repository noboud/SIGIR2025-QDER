"""
Model architectures for QDER.

This package contains:
- Base model classes and interfaces
- QDER model implementations
- Model components (text embeddings, attention layers, etc.)
- Model factory for creating models from arguments
- Ablation study variants
"""

from .base_model import BaseQDERModel
from .text_embedding import TextEmbedding
from .qder_model import QDERModel
from .qder_ablation import QDERAblation
from .model_factory import create_model, get_model_class

__all__ = [
    'BaseQDERModel',
    'TextEmbedding',
    'QDERModel',
    'QDERAblation',
    'create_model',
    'get_model_class'
]