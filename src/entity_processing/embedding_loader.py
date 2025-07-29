"""
Entity embedding loading and processing utilities.

This module provides functionality to load and manage entity embeddings
from various formats, with support for large-scale embedding files and
efficient batch processing.
"""

import json
import gzip
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingLoadingError(Exception):
    """Custom exception for embedding loading errors."""
    pass


@dataclass
class EntityInfo:
    """
    Container for entity information including embeddings.

    Attributes:
        entity_id: Unique identifier for the entity
        entity_name: Human-readable name/title
        embedding: Numerical embedding vector
        score: Optional relevance or confidence score
        metadata: Additional metadata dictionary
    """
    entity_id: str
    entity_name: str
    embedding: List[float]
    score: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'embedding': self.embedding,
            'score': self.score,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityInfo':
        """Create EntityInfo from dictionary."""
        return cls(
            entity_id=data['entity_id'],
            entity_name=data['entity_name'],
            embedding=data['embedding'],
            score=data.get('score', 1.0),
            metadata=data.get('metadata', {})
        )

    def truncate_embedding(self, max_dim: int) -> 'EntityInfo':
        """
        Create a copy with truncated embedding.

        Args:
            max_dim: Maximum embedding dimensions to keep

        Returns:
            New EntityInfo with truncated embedding
        """
        truncated_embedding = self.embedding[:max_dim] if len(self.embedding) > max_dim else self.embedding
        return EntityInfo(
            entity_id=self.entity_id,
            entity_name=self.entity_name,
            embedding=truncated_embedding,
            score=self.score,
            metadata=self.metadata.copy()
        )


class EntityEmbeddingLoader:
    """
    Loader for entity embeddings from various formats.

    Supports loading from compressed and uncompressed files with
    flexible parsing and error handling.
    """

    def __init__(self, embedding_dim: int = 300, normalize_embeddings: bool = False):
        """
        Initialize embedding loader.

        Args:
            embedding_dim: Target embedding dimension
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings

    def load_embeddings(self,
                        embedding_file: Union[str, Path],
                        total_count: Optional[int] = None,
                        entity_filter: Optional[set] = None) -> Dict[str, EntityInfo]:
        """
        Load entity embeddings from file.

        Args:
            embedding_file: Path to embedding file (supports .gz)
            total_count: Total number of embeddings for progress bar
            entity_filter: Set of entity IDs to load (load all if None)

        Returns:
            Dictionary mapping entity_id to EntityInfo

        Raises:
            EmbeddingLoadingError: If file cannot be loaded
        """
        embeddings = {}
        embedding_file = Path(embedding_file)

        if not embedding_file.exists():
            raise EmbeddingLoadingError(f"Embedding file not found: {embedding_file}")

        try:
            # Determine file opener
            opener = gzip.open if embedding_file.suffix == '.gz' else open

            logger.info(f"Loading embeddings from {embedding_file}")

            with opener(embedding_file, 'rt') as f:
                progress_bar = tqdm(f, total=total_count, desc="Loading embeddings") if total_count else f

                for line_num, line in enumerate(progress_bar, 1):
                    try:
                        data = json.loads(line.strip())
                        entity_info = self._parse_embedding_line(data)

                        # Apply entity filter if provided
                        if entity_filter and entity_info.entity_id not in entity_filter:
                            continue

                        # Truncate embedding if needed
                        if len(entity_info.embedding) > self.embedding_dim:
                            entity_info = entity_info.truncate_embedding(self.embedding_dim)

                        # Normalize if requested
                        if self.normalize_embeddings:
                            entity_info = self._normalize_embedding(entity_info)

                        embeddings[entity_info.entity_id] = entity_info

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Skipping malformed line {line_num}: {e}")
                        continue

        except Exception as e:
            raise EmbeddingLoadingError(f"Error loading embeddings from {embedding_file}: {e}")

        logger.info(f"Loaded {len(embeddings)} entity embeddings")
        return embeddings

    def load_with_mappings(self,
                           embedding_file: Union[str, Path],
                           name_mapping: Optional[Dict[str, str]] = None,
                           id_mapping: Optional[Dict[str, str]] = None,
                           total_count: Optional[int] = None) -> Tuple[Dict[str, EntityInfo], List[str]]:
        """
        Load embeddings with entity name/ID mappings.

        Args:
            embedding_file: Path to embedding file
            name_mapping: Mapping from entity names to target IDs
            id_mapping: Mapping from source IDs to target IDs
            total_count: Total embeddings for progress bar

        Returns:
            Tuple of (loaded_embeddings, not_found_entities)
        """
        embeddings = {}
        not_found = []

        # Load all embeddings first
        all_embeddings = self.load_embeddings(embedding_file, total_count)

        # Apply mappings
        for entity_id, entity_info in all_embeddings.items():
            mapped_id = None

            # Try name mapping first
            if name_mapping and entity_info.entity_name in name_mapping:
                mapped_id = name_mapping[entity_info.entity_name]
            # Then try ID mapping
            elif id_mapping and entity_id in id_mapping:
                mapped_id = id_mapping[entity_id]

            if mapped_id:
                # Update entity ID to mapped ID
                new_entity_info = EntityInfo(
                    entity_id=mapped_id,
                    entity_name=entity_info.entity_name,
                    embedding=entity_info.embedding,
                    score=entity_info.score,
                    metadata=entity_info.metadata
                )
                embeddings[mapped_id] = new_entity_info
            else:
                not_found.append(json.dumps({
                    'entity_name': entity_info.entity_name,
                    'entity_id': entity_id
                }))

        logger.info(f"Mapped {len(embeddings)} embeddings, {len(not_found)} not found")
        return embeddings, not_found

    def _parse_embedding_line(self, data: Dict[str, Any]) -> EntityInfo:
        """
        Parse a single embedding line from JSON data.

        Args:
            data: JSON data dictionary

        Returns:
            EntityInfo object

        Raises:
            KeyError: If required fields are missing
            ValueError: If embedding is invalid
        """
        try:
            entity_id = data['entity_id']
            entity_name = data['entity_name']
            embedding = data['embedding']

            # Validate embedding
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Invalid embedding format")

            # Ensure all embedding values are numbers
            embedding = [float(x) for x in embedding]

            return EntityInfo(
                entity_id=str(entity_id),
                entity_name=str(entity_name),
                embedding=embedding,
                score=data.get('score', 1.0),
                metadata=data.get('metadata', {})
            )

        except KeyError as e:
            raise KeyError(f"Missing required field: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid embedding data: {e}")

    def _normalize_embedding(self, entity_info: EntityInfo) -> EntityInfo:
        """
        L2-normalize an embedding.

        Args:
            entity_info: EntityInfo with embedding to normalize

        Returns:
            EntityInfo with normalized embedding
        """
        embedding = np.array(entity_info.embedding)
        norm = np.linalg.norm(embedding)

        if norm > 0:
            normalized_embedding = (embedding / norm).tolist()
        else:
            normalized_embedding = entity_info.embedding

        return EntityInfo(
            entity_id=entity_info.entity_id,
            entity_name=entity_info.entity_name,
            embedding=normalized_embedding,
            score=entity_info.score,
            metadata=entity_info.metadata
        )


def load_entity_mappings(mapping_file: Union[str, Path],
                         key_col: int = 0,
                         value_col: int = 1,
                         delimiter: str = '\t') -> Dict[str, str]:
    """
    Load entity ID/name mappings from TSV file.

    Args:
        mapping_file: Path to mapping file
        key_col: Column index for keys
        value_col: Column index for values
        delimiter: Field delimiter

    Returns:
        Dictionary mapping keys to values

    Raises:
        EmbeddingLoadingError: If file cannot be loaded
    """
    mappings = {}

    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading mappings"), 1):
                try:
                    parts = line.strip().split(delimiter)
                    if len(parts) > max(key_col, value_col):
                        key = parts[key_col].strip()
                        value = parts[value_col].strip()
                        mappings[key] = value
                    else:
                        logger.warning(f"Skipping malformed line {line_num}: insufficient columns")
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        raise EmbeddingLoadingError(f"Mapping file not found: {mapping_file}")
    except Exception as e:
        raise EmbeddingLoadingError(f"Error loading mappings from {mapping_file}: {e}")

    logger.info(f"Loaded {len(mappings)} mappings from {mapping_file}")
    return mappings


def get_entity_embeddings_batch(entity_ids: List[str],
                                embeddings: Dict[str, EntityInfo],
                                default_embedding: Optional[List[float]] = None) -> List[List[float]]:
    """
    Get embeddings for a batch of entity IDs.

    Args:
        entity_ids: List of entity IDs
        embeddings: Dictionary of loaded embeddings
        default_embedding: Default embedding for missing entities

    Returns:
        List of embedding vectors
    """
    batch_embeddings = []

    for entity_id in entity_ids:
        if entity_id in embeddings:
            batch_embeddings.append(embeddings[entity_id].embedding)
        elif default_embedding is not None:
            batch_embeddings.append(default_embedding)
            logger.debug(f"Using default embedding for missing entity: {entity_id}")
        else:
            logger.warning(f"Entity not found and no default provided: {entity_id}")

    return batch_embeddings


def filter_embeddings_by_entities(embeddings: Dict[str, EntityInfo],
                                  entity_ids: set) -> Dict[str, EntityInfo]:
    """
    Filter embeddings to only include specified entities.

    Args:
        embeddings: Dictionary of all embeddings
        entity_ids: Set of entity IDs to keep

    Returns:
        Filtered embeddings dictionary
    """
    filtered = {eid: emb for eid, emb in embeddings.items() if eid in entity_ids}
    logger.info(f"Filtered embeddings from {len(embeddings)} to {len(filtered)} entities")
    return filtered


def save_embeddings(embeddings: Dict[str, EntityInfo],
                    output_file: Union[str, Path],
                    compress: bool = True) -> None:
    """
    Save embeddings to file.

    Args:
        embeddings: Dictionary of embeddings to save
        output_file: Output file path
        compress: Whether to compress output with gzip

    Raises:
        EmbeddingLoadingError: If file cannot be written
    """
    output_path = Path(output_file)

    try:
        opener = gzip.open if compress else open
        mode = 'wt' if compress else 'w'

        with opener(output_path, mode) as f:
            for entity_info in tqdm(embeddings.values(), desc="Saving embeddings"):
                json.dump(entity_info.to_dict(), f)
                f.write('\n')

        logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")

    except Exception as e:
        raise EmbeddingLoadingError(f"Error saving embeddings to {output_path}: {e}")


# Legacy function for backward compatibility
def load_entity_embeddings(embedding_file: str,
                           name2id: Optional[Dict[str, str]] = None,
                           wiki2car: Optional[Dict[str, str]] = None,
                           embedding_dim: int = 300) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    Legacy function for loading entity embeddings.

    Args:
        embedding_file: Path to embedding file
        name2id: Name to ID mapping
        wiki2car: Wikipedia to CAR ID mapping
        embedding_dim: Embedding dimension

    Returns:
        Tuple of (embeddings_dict, not_found_list)

    Note:
        This function is deprecated. Use EntityEmbeddingLoader class instead.
    """
    import warnings
    warnings.warn(
        "load_entity_embeddings is deprecated. Use EntityEmbeddingLoader class instead.",
        DeprecationWarning,
        stacklevel=2
    )

    loader = EntityEmbeddingLoader(embedding_dim=embedding_dim)

    if name2id or wiki2car:
        embeddings, not_found = loader.load_with_mappings(
            embedding_file, name2id, wiki2car
        )
        # Convert to legacy format
        legacy_embeddings = {eid: info.embedding for eid, info in embeddings.items()}
        return legacy_embeddings, not_found
    else:
        embeddings = loader.load_embeddings(embedding_file)
        legacy_embeddings = {eid: info.embedding for eid, info in embeddings.items()}
        return legacy_embeddings, []