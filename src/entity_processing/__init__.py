"""
Entity processing components for CADR models.

This package contains:
- Entity linking using WAT (Wikipedia Annotation Tool)
- Entity embedding loading and processing
- Entity ranking creation from document rankings
- Entity-centric data processing utilities
"""

# Import entity linking from the existing data module
from ..data.entity_linking import (
    WATEntityLinker,
    WATAnnotation,
    link_documents_from_file,
    load_documents_for_linking,
    write_entity_results
)

# Import embedding functionality from our new module
from .embedding_loader import (
    EntityEmbeddingLoader,
    EntityInfo,
    EmbeddingLoadingError,
    load_entity_mappings,
    get_entity_embeddings_batch,
    filter_embeddings_by_entities,
    save_embeddings,
    # Legacy function
    load_entity_embeddings
)

# Import ranking functionality from our new module
from .entity_ranking import (
    EntityRanker,
    EntityRankingError,
    create_entity_rankings,
    load_document_run,
    load_document_entities,
    write_entity_rankings_to_file,
    compute_entity_statistics,
    # Score transformation functions
    sigmoid,
    log_scale,
    min_max_scaling,
    z_score_normalization,
    # Legacy functions
    compute_entity_rankings_for_all_queries,
    update_entity_scores_for_query
)

__all__ = [
    # Entity linking (from existing data module)
    'WATEntityLinker',
    'WATAnnotation',
    'link_documents_from_file',
    'load_documents_for_linking',
    'write_entity_results',

    # Embedding loading
    'EntityEmbeddingLoader',
    'EntityInfo',
    'EmbeddingLoadingError',
    'load_entity_mappings',
    'get_entity_embeddings_batch',
    'filter_embeddings_by_entities',
    'save_embeddings',
    'load_entity_embeddings',  # Legacy

    # Entity ranking
    'EntityRanker',
    'EntityRankingError',
    'create_entity_rankings',
    'load_document_run',
    'load_document_entities',
    'write_entity_rankings_to_file',
    'compute_entity_statistics',
    'sigmoid',
    'log_scale',
    'min_max_scaling',
    'z_score_normalization',
    'compute_entity_rankings_for_all_queries',  # Legacy
    'update_entity_scores_for_query',  # Legacy
]

__version__ = '1.0.0'