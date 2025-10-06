#!/usr/bin/env python3
"""
Create reranking data from document corpus, entity embeddings, and run files.

This script processes raw documents with entity annotations and creates
query-document pairs with relevance labels for training QDER models.
"""

import sys
import os
from statistics import mean

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import argparse
import collections
from tqdm import tqdm
from typing import Dict, Tuple, List, Set, Any
from dataclasses import dataclass
from pathlib import Path
import logging

from src.utils import setup_logging, ensure_dir_exists
from src.entity_processing import EntityEmbeddingLoader, EntityInfo

logger = logging.getLogger(__name__)


@dataclass
class EntityData:
    """Container for entity information with metadata."""
    entity_id: str
    entity_name: str
    embedding: List[float]
    score: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """Processes documents and creates reranking data."""

    def __init__(self, embedding_dim: int = 300):
        """
        Initialize document processor.

        Args:
            embedding_dim: Target embedding dimension
        """
        self.embedding_dim = embedding_dim
        self.entity_loader = EntityEmbeddingLoader(embedding_dim)

    def load_embeddings(self, embedding_file: str,
                        name_mapping: Dict[str, str] = None,
                        id_mapping: Dict[str, str] = None) -> Tuple[Dict[str, EntityInfo], List[str]]:
        """
        Load entity embeddings with optional mappings.

        Args:
            embedding_file: Path to embedding file
            name_mapping: Entity name to ID mapping
            id_mapping: Entity ID to ID mapping

        Returns:
            Tuple of (embeddings_dict, not_found_entities)
        """
        logger.info('Loading embeddings...')

        if name_mapping or id_mapping:
            embeddings, not_found = self.entity_loader.load_with_mappings(
                embedding_file, name_mapping, id_mapping
            )
        else:
            embeddings = self.entity_loader.load_embeddings(embedding_file)
            not_found = []

        logger.info(f'Loaded {len(embeddings)} embeddings')
        return embeddings, not_found

    def load_docs(self, docs_file: str) -> Dict[str, Tuple[List[str], str]]:
        """
        Load documents with entity annotations.

        Args:
            docs_file: Path to documents JSONL file

        Returns:
            Dictionary mapping doc_id to (entities, text)
        """
        docs = {}
        logger.info('Loading documents...')

        try:
            with open(docs_file, 'r') as f:
                for line in tqdm(f, desc="Loading documents"):
                    try:
                        doc_data = json.loads(line.strip())
                        docs[doc_data['doc_id']] = (doc_data['entities'], doc_data['text'])
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed document: {e}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Document file not found: {docs_file}")

        logger.info(f'Loaded {len(docs)} documents')
        return docs

    def get_entity_centric_embeddings(self,
                                      entity_ids: List[str],
                                      entity_scores: Dict[str, float],
                                      entity_info: Dict[str, EntityInfo]) -> List[Dict[str, Any]]:
        """
        Get entity-centric embeddings with scores.

        Args:
            entity_ids: List of entity IDs
            entity_scores: Mapping of entity ID to score
            entity_info: Mapping of entity ID to EntityInfo

        Returns:
            List of entity embedding dictionaries
        """
        embeddings = []

        for entity_id in entity_ids:
            entity_id = str(entity_id)
            if entity_id in entity_info and entity_id in entity_scores:
                entity = entity_info[entity_id]
                score = entity_scores[entity_id]

                # Apply score weighting to embedding
                weighted_embedding = [score * val for val in entity.embedding]

                embeddings.append({
                    'entity_id': entity_id,
                    'entity_name': entity.entity_name,
                    'embedding': weighted_embedding,
                    'score': score,
                    'metadata': entity.metadata
                })

        return embeddings if embeddings else []

    def get_docs_by_relevance(self,
                              docs: Dict[str, Tuple[List[str], str]],
                              qrels: Dict[str, int],
                              query_entities: Dict[str, float],
                              entity_info: Dict[str, EntityInfo],
                              positive: bool,
                              query_docs: Set[str],
                              doc_scores: Dict[str, float]) -> Dict[str, Tuple[str, float, List[Dict[str, Any]]]]:
        """
        Get documents filtered by relevance.

        Args:
            docs: Document data
            qrels: Relevance judgments
            query_entities: Query entity scores
            entity_info: Entity information
            positive: Whether to get positive or negative documents
            query_docs: Set of valid document IDs
            doc_scores: Document retrieval scores

        Returns:
            Dictionary mapping doc_id to (text, score, entity_embeddings)
        """
        results = {}
        for doc_id in query_docs:
            if doc_id not in docs or doc_id not in doc_scores:
                continue

            is_positive = doc_id in qrels and qrels[doc_id] >= 1
            if is_positive != positive:
                continue

            doc_entities, doc_text = docs[doc_id]
            doc_ent_embeddings = self.get_entity_centric_embeddings(
                doc_entities, query_entities, entity_info
            )

            if doc_ent_embeddings:
                results[doc_id] = (doc_text, doc_scores[doc_id], doc_ent_embeddings)

        return results


class DataCreator:
    """Creates training/test data from processed components."""

    def __init__(self, processor: DocumentProcessor):
        """
        Initialize data creator.

        Args:
            processor: Document processor instance
        """
        self.processor = processor

    @staticmethod
    def read_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
        """Read TREC format qrels file."""
        qrels = collections.defaultdict(dict)

        with open(qrels_file, 'r') as f:
            for line in f:
                try:
                    query_id, _, object_id, relevance = line.strip().split()
                    qrels[query_id][object_id] = int(relevance)
                except ValueError:
                    logger.warning(f"Skipping malformed qrels line: {line.strip()}")
                    continue

        logger.info(f'Loaded qrels for {len(qrels)} queries')
        return qrels

    @staticmethod
    def read_run(run_file: str) -> Dict[str, Dict[str, float]]:
        """Read TREC format run file."""
        run = collections.defaultdict(dict)

        with open(run_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    query_id, object_id, score = parts[0], parts[2], float(parts[4])

                    if object_id not in run[query_id]:
                        run[query_id][object_id] = score
                except (ValueError, IndexError):
                    logger.warning(f"Skipping malformed run line: {line.strip()}")
                    continue

        logger.info(f'Loaded run data for {len(run)} queries')
        return run

    @staticmethod
    def load_fold_queries(fold_file: str, fold: int, testing: bool=False):
        if fold_file:
            with open(fold_file, 'r') as f:
                fold = json.load(f)[str(fold)]
                return fold["testing"] if testing else fold["training"]
        else:
            return []

    @staticmethod
    def load_queries(queries_file: str, fold_queries: list[str]=None) -> Dict[str, str]:
        """Load queries from TSV file."""
        queries = {}

        with open(queries_file, 'r') as f:
            for line in f:
                try:
                    query_id, query_text = line.strip().split('\t', 1)
                    if fold_queries and not str(query_id) in fold_queries:
                        continue
                    queries[query_id] = query_text
                except ValueError:
                    logger.warning(f"Skipping malformed query line: {line.strip()}")
                    continue

        logger.info(f'Loaded {len(queries)} queries')
        return queries

    def create_data(self,
                    queries: Dict[str, str],
                    docs: Dict[str, Tuple[List[str], str]],
                    doc_qrels: Dict[str, Dict[str, int]],
                    doc_run: Dict[str, Dict[str, float]],
                    entity_run: Dict[str, Dict[str, float]],
                    entity_info: Dict[str, EntityInfo],
                    k: int,
                    train: bool,
                    balance: bool,
                    save_path: str) -> int:
        """
        Create complete dataset.

        Args:
            queries: Query text mapping
            docs: Document data
            doc_qrels: Document relevance judgments
            doc_run: Document run data
            entity_run: Entity run data
            entity_info: Entity information
            k: Number of top entities to use
            train: Whether creating training data
            balance: Whether to balance positive/negative examples
            save_path: Output file path

        Returns:
            Number of examples created
        """
        ensure_dir_exists(save_path)
        examples_count = 0
        positives_lengths = []

        with open(save_path, 'w') as f:
            for query_id, query_text in tqdm(queries.items(), desc="Processing queries"):
                if not all(query_id in x for x in [doc_run, entity_run, doc_qrels]):
                    continue

                query_docs = doc_run[query_id]
                query_entities = entity_run[query_id]
                qrels = doc_qrels[query_id]

                # Get query entity embeddings (top-k)
                top_entities = dict(list(query_entities.items())[:k])
                query_ent_emb = self.processor.get_entity_centric_embeddings(
                    list(top_entities.keys()), top_entities, entity_info
                )

                if not query_ent_emb:
                    continue

                # Get positive and negative documents
                doc_source = set(qrels.keys()) if train else set(query_docs.keys())

                pos_docs = self.processor.get_docs_by_relevance(
                    docs, qrels, query_entities, entity_info,
                    True, doc_source, query_docs
                )

                positives_lengths.append(len(pos_docs))

                neg_docs = self.processor.get_docs_by_relevance(
                    docs, qrels, query_entities, entity_info,
                    False, set(query_docs.keys()), query_docs
                )

                # Balance data if requested
                if balance:
                    n = min(len(pos_docs), len(neg_docs))
                    pos_docs = dict(list(pos_docs.items())[:n])
                    neg_docs = dict(list(neg_docs.items())[:n])

                # Write data
                for label, doc_set in [(1, pos_docs), (0, neg_docs)]:
                    for doc_id, (doc_text, doc_score, doc_ent_emb) in doc_set.items():
                        data_point = {
                            'query': query_text,
                            'query_id': query_id,
                            'query_ent_emb': query_ent_emb,
                            'doc_id': doc_id,
                            'doc': doc_text,
                            'doc_score': doc_score,
                            'doc_ent_emb': doc_ent_emb,
                            'label': label
                        }

                        json.dump(data_point, f, ensure_ascii=False)
                        f.write('\n')
                        examples_count += 1

        positives_count = collections.Counter(positives_lengths)
        positives_string = "\n".join([f"{cnt}x {key}" for key, cnt in positives_count.most_common()])
        logger.debug(f'Positives per query (as a counter) (mean: {mean(positives_lengths)}, min: {min(positives_lengths)}, max: {max(positives_lengths)}):\n{positives_string}')

        logger.info(f'Created {examples_count} examples')
        return examples_count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create train/test data with enhanced entity information"
    )

    # Required arguments
    parser.add_argument("--queries", required=True, type=str, help="Queries TSV file")
    parser.add_argument("--docs", required=True, type=str, help="Document JSONL file")
    parser.add_argument("--qrels", required=True, type=str, help="Document qrels file")
    parser.add_argument("--doc-run", required=True, type=str, help="Document run file")
    parser.add_argument("--entity-run", required=True, type=str, help="Entity run file")
    parser.add_argument("--embeddings", required=True, type=str, help="Entity embeddings file")
    parser.add_argument("--save", required=True, type=str, help="Output file path")

    # Optional arguments
    parser.add_argument("--k", default=20, type=int, help="Number of expansion entities (default: 20)")
    parser.add_argument("--embedding-dim", default=300, type=int, help="Embedding dimension (default: 300)")
    parser.add_argument('--train', action='store_true', help='Create training data')
    parser.add_argument('--balance', action='store_true', help='Balance positive/negative examples')

    # Entity mapping files (optional)
    parser.add_argument("--name-mapping", type=str, help="Entity name to ID mapping file")
    parser.add_argument("--id-mapping", type=str, help="Entity ID to ID mapping file")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Experimentation
    parser.add_argument("--folds", type=str)
    parser.add_argument("--fold-index", type=int)

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info(f"Creating {'training' if args.train else 'test'} data")
    logger.info(f"Data will be {'balanced' if args.balance else 'unbalanced'}")

    if args.folds and args.fold_index is None:
        logger.error(f"Folds given without index, ignoring folds...")

    # Initialize components
    processor = DocumentProcessor(embedding_dim=args.embedding_dim)
    creator = DataCreator(processor)

    # Load entity mappings if provided
    name_mapping = None
    id_mapping = None

    if args.name_mapping:
        logger.info("Loading name mapping...")
        from src.entity_processing import load_entity_mappings
        name_mapping = load_entity_mappings(args.name_mapping, key_col=1, value_col=0)

    if args.id_mapping:
        logger.info("Loading ID mapping...")
        from src.entity_processing import load_entity_mappings
        id_mapping = load_entity_mappings(args.id_mapping)

    # Load all required data
    logger.info("Loading data files...")

    fold_queries = creator.load_fold_queries(
        args.folds,
        args.fold_index,
        testing=not args.train
    ) if args.folds and not args.fold_index is None else []
    queries = creator.load_queries(args.queries, fold_queries=fold_queries)
    docs = processor.load_docs(args.docs)
    qrels = creator.read_qrels(args.qrels)
    doc_run = creator.read_run(args.doc_run)
    entity_run = creator.read_run(args.entity_run)

    entity_info, not_found = processor.load_embeddings(
        args.embeddings, name_mapping, id_mapping
    )

    # Save not found entities for debugging
    if not_found:
        not_found_path = Path(args.save).parent / "not_found_entities.jsonl"
        with open(not_found_path, 'w') as f:
            for entity in not_found:
                f.write(f"{entity}\n")
        logger.info(f"Saved {len(not_found)} not found entities to {not_found_path}")

    # Create dataset
    logger.info("Creating dataset...")
    examples_count = creator.create_data(
        queries=queries,
        docs=docs,
        doc_qrels=qrels,
        doc_run=doc_run,
        entity_run=entity_run,
        entity_info=entity_info,
        k=args.k,
        train=args.train,
        balance=args.balance,
        save_path=args.save
    )

    logger.info(f"Dataset creation complete! Created {examples_count} examples")
    logger.info(f"Output saved to: {args.save}")


if __name__ == '__main__':
    main()