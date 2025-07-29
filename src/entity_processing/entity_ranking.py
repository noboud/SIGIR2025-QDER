"""
Entity ranking creation from document rankings.

This module provides functionality to create entity rankings from document rankings
by aggregating document scores for entities contained in those documents. It includes
various score normalization and aggregation strategies.
"""

import json
import logging
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EntityRankingError(Exception):
    """Custom exception for entity ranking errors."""
    pass


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Apply sigmoid function to input.

    Args:
        x: Input value(s)

    Returns:
        Sigmoid-transformed value(s)
    """
    return 1 / (1 + np.exp(-np.array(x)))


def log_scale(scores: List[float]) -> List[float]:
    """
    Apply logarithmic scaling to scores.

    Args:
        scores: List of scores to scale

    Returns:
        Log-scaled scores
    """
    return np.log1p(scores).tolist()  # log(1 + scores)


def min_max_scaling(scores: List[float]) -> List[float]:
    """
    Apply min-max normalization to scores.

    Args:
        scores: List of scores to normalize

    Returns:
        Min-max normalized scores
    """
    if not scores:
        return scores

    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()

    if max_score == min_score:
        return [1.0] * len(scores)  # All scores are the same

    return ((scores_array - min_score) / (max_score - min_score)).tolist()


def z_score_normalization(scores: List[float]) -> List[float]:
    """
    Apply z-score normalization to scores.

    Args:
        scores: List of scores to normalize

    Returns:
        Z-score normalized scores
    """
    if not scores:
        return scores

    scores_array = np.array(scores)
    mean_score = scores_array.mean()
    std_score = scores_array.std()

    if std_score == 0:
        return [0.0] * len(scores)  # All scores are the same

    return ((scores_array - mean_score) / std_score).tolist()


class EntityRanker:
    """
    Creates entity rankings from document rankings.

    Aggregates document scores for entities and applies various
    normalization strategies to create entity rankings.
    """

    def __init__(self,
                 aggregation_method: str = 'sum',
                 normalization_method: str = 'min_max',
                 log_transform: bool = True):
        """
        Initialize entity ranker.

        Args:
            aggregation_method: Method to aggregate scores ('sum', 'mean', 'max')
            normalization_method: Normalization method ('min_max', 'z_score', 'sigmoid', 'none')
            log_transform: Whether to apply log transformation before normalization
        """
        self.aggregation_method = aggregation_method
        self.normalization_method = normalization_method
        self.log_transform = log_transform

        # Validate parameters
        valid_aggregation = {'sum', 'mean', 'max'}
        valid_normalization = {'min_max', 'z_score', 'sigmoid', 'none'}

        if aggregation_method not in valid_aggregation:
            raise EntityRankingError(f"Invalid aggregation method: {aggregation_method}")
        if normalization_method not in valid_normalization:
            raise EntityRankingError(f"Invalid normalization method: {normalization_method}")

    def create_entity_rankings(self,
                               doc_run: Dict[str, Dict[str, float]],
                               docs: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Create entity rankings for all queries.

        Args:
            doc_run: Document run data {query_id: {doc_id: score}}
            docs: Document data {doc_id: [entity_ids]}

        Returns:
            Entity rankings {query_id: [(entity_id, score), ...]}
        """
        entity_rankings = {}

        logger.info(f"Creating entity rankings for {len(doc_run)} queries")

        for query_id, doc_scores in tqdm(doc_run.items(), desc="Processing queries"):
            entity_scores = self._aggregate_entity_scores(doc_scores, docs)

            if entity_scores:
                # Apply normalization
                normalized_scores = self._normalize_scores(entity_scores)

                # Sort by score (descending)
                ranked_entities = sorted(
                    normalized_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                entity_rankings[query_id] = ranked_entities
            else:
                entity_rankings[query_id] = []
                logger.warning(f"No entities found for query {query_id}")

        logger.info(f"Created entity rankings for {len(entity_rankings)} queries")
        return entity_rankings

    def _aggregate_entity_scores(self,
                                 doc_scores: Dict[str, float],
                                 docs: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Aggregate document scores for entities.

        Args:
            doc_scores: Document scores for a query
            docs: Document-entity mappings

        Returns:
            Aggregated entity scores
        """
        entity_scores = defaultdict(list)

        # Collect scores for each entity
        for doc_id, doc_score in doc_scores.items():
            if doc_id in docs:
                for entity_id in docs[doc_id]:
                    entity_scores[entity_id].append(doc_score)

        # Aggregate scores
        aggregated_scores = {}
        for entity_id, scores in entity_scores.items():
            if self.aggregation_method == 'sum':
                aggregated_scores[entity_id] = sum(scores)
            elif self.aggregation_method == 'mean':
                aggregated_scores[entity_id] = sum(scores) / len(scores)
            elif self.aggregation_method == 'max':
                aggregated_scores[entity_id] = max(scores)

        return aggregated_scores

    def _normalize_scores(self, entity_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize entity scores.

        Args:
            entity_scores: Raw entity scores

        Returns:
            Normalized entity scores
        """
        if not entity_scores:
            return entity_scores

        entities = list(entity_scores.keys())
        scores = list(entity_scores.values())

        # Apply log transformation if requested
        if self.log_transform:
            scores = log_scale(scores)

        # Apply normalization
        if self.normalization_method == 'min_max':
            normalized_scores = min_max_scaling(scores)
        elif self.normalization_method == 'z_score':
            normalized_scores = z_score_normalization(scores)
        elif self.normalization_method == 'sigmoid':
            normalized_scores = sigmoid(scores).tolist()
        else:  # 'none'
            normalized_scores = scores

        return dict(zip(entities, normalized_scores))


def load_document_run(run_file: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load document run file in TREC format.

    Args:
        run_file: Path to run file

    Returns:
        Document run data {query_id: {doc_id: score}}

    Raises:
        EntityRankingError: If file cannot be loaded
    """
    run_data = defaultdict(dict)

    try:
        with open(run_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        query_id = parts[0]
                        doc_id = parts[2]
                        score = float(parts[4])

                        # Keep only the highest score for each doc (in case of duplicates)
                        if doc_id not in run_data[query_id] or score > run_data[query_id][doc_id]:
                            run_data[query_id][doc_id] = score
                    else:
                        logger.warning(f"Skipping malformed line {line_num}: insufficient fields")

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        raise EntityRankingError(f"Run file not found: {run_file}")
    except Exception as e:
        raise EntityRankingError(f"Error loading run file {run_file}: {e}")

    logger.info(f"Loaded document run with {len(run_data)} queries")
    return dict(run_data)


def load_document_entities(doc_file: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Load document-entity mappings from JSONL file.

    Args:
        doc_file: Path to document file

    Returns:
        Document-entity mappings {doc_id: [entity_ids]}

    Raises:
        EntityRankingError: If file cannot be loaded
    """
    docs = {}

    try:
        with open(doc_file, 'r') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading documents"), 1):
                try:
                    doc_data = json.loads(line.strip())
                    doc_id = doc_data['doc_id']
                    entities = doc_data.get('entities', [])

                    # Ensure entities are strings
                    entities = [str(entity) for entity in entities]
                    docs[doc_id] = entities

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        raise EntityRankingError(f"Document file not found: {doc_file}")
    except Exception as e:
        raise EntityRankingError(f"Error loading document file {doc_file}: {e}")

    logger.info(f"Loaded {len(docs)} documents with entity mappings")
    return docs


def write_entity_rankings_to_file(entity_rankings: Dict[str, List[Tuple[str, float]]],
                                  output_file: Union[str, Path],
                                  run_name: str = "EntityRanking") -> None:
    """
    Write entity rankings to TREC format file.

    Args:
        entity_rankings: Entity rankings {query_id: [(entity_id, score), ...]}
        output_file: Output file path
        run_name: Run name for TREC format

    Raises:
        EntityRankingError: If file cannot be written
    """
    try:
        with open(output_file, 'w') as f:
            for query_id, ranked_entities in entity_rankings.items():
                for rank, (entity_id, score) in enumerate(ranked_entities, 1):
                    # TREC format: query_id Q0 doc_id rank score run_name
                    f.write(f"{query_id} Q0 {entity_id} {rank} {score:.6f} {run_name}\n")

        logger.info(f"Wrote entity rankings to {output_file}")

    except Exception as e:
        raise EntityRankingError(f"Error writing entity rankings to {output_file}: {e}")


def compute_entity_statistics(entity_rankings: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Union[int, float]]:
    """
    Compute statistics about entity rankings.

    Args:
        entity_rankings: Entity rankings data

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_queries': len(entity_rankings),
        'total_entities': 0,
        'avg_entities_per_query': 0.0,
        'min_entities_per_query': float('inf'),
        'max_entities_per_query': 0,
        'queries_with_no_entities': 0
    }

    if not entity_rankings:
        return stats

    entity_counts = []
    all_entities = set()

    for query_id, entities in entity_rankings.items():
        count = len(entities)
        entity_counts.append(count)

        if count == 0:
            stats['queries_with_no_entities'] += 1
        else:
            stats['min_entities_per_query'] = min(stats['min_entities_per_query'], count)
            stats['max_entities_per_query'] = max(stats['max_entities_per_query'], count)

        # Collect unique entities
        for entity_id, _ in entities:
            all_entities.add(entity_id)

    stats['total_entities'] = len(all_entities)
    stats['avg_entities_per_query'] = sum(entity_counts) / len(entity_counts)

    if stats['min_entities_per_query'] == float('inf'):
        stats['min_entities_per_query'] = 0

    return stats


# Legacy functions for backward compatibility
def update_entity_scores_for_query(doc_run_for_query: Dict[str, float],
                                   docs: Dict[str, List[str]],
                                   entity_scores: Dict[str, float]) -> None:
    """
    Legacy function for updating entity scores.

    Note:
        This function is deprecated. Use EntityRanker class instead.
    """
    import warnings
    warnings.warn(
        "update_entity_scores_for_query is deprecated. Use EntityRanker class instead.",
        DeprecationWarning,
        stacklevel=2
    )

    for doc_id, doc_score in doc_run_for_query.items():
        if doc_id in docs:
            for entity_id in docs[doc_id]:
                entity_scores[entity_id] += doc_score


def compute_entity_rankings_for_all_queries(doc_run: Dict[str, Dict[str, float]],
                                            docs: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Legacy function for computing entity rankings.

    Note:
        This function is deprecated. Use EntityRanker class instead.
    """
    import warnings
    warnings.warn(
        "compute_entity_rankings_for_all_queries is deprecated. Use EntityRanker class instead.",
        DeprecationWarning,
        stacklevel=2
    )

    ranker = EntityRanker()
    return ranker.create_entity_rankings(doc_run, docs)


def create_entity_rankings(doc_run_file: Union[str, Path],
                           doc_entities_file: Union[str, Path],
                           output_file: Union[str, Path],
                           aggregation_method: str = 'sum',
                           normalization_method: str = 'min_max') -> Dict[str, Union[int, float]]:
    """
    Complete pipeline for creating entity rankings from document rankings.

    Args:
        doc_run_file: Path to document run file
        doc_entities_file: Path to document-entities file
        output_file: Path to output entity ranking file
        aggregation_method: Score aggregation method
        normalization_method: Score normalization method

    Returns:
        Statistics about the created rankings
    """
    # Load data
    logger.info("Loading document run file...")
    doc_run = load_document_run(doc_run_file)

    logger.info("Loading document-entity mappings...")
    docs = load_document_entities(doc_entities_file)

    # Create rankings
    logger.info("Creating entity rankings...")
    ranker = EntityRanker(
        aggregation_method=aggregation_method,
        normalization_method=normalization_method
    )
    entity_rankings = ranker.create_entity_rankings(doc_run, docs)

    # Write output
    logger.info("Writing entity rankings...")
    write_entity_rankings_to_file(entity_rankings, output_file)

    # Compute and return statistics
    stats = compute_entity_statistics(entity_rankings)
    logger.info(f"Entity ranking statistics: {stats}")

    return stats