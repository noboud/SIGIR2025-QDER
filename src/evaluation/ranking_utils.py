"""
Ranking utilities for TREC evaluation.

Contains your original utils.py save_trec function and related utilities.
"""

import os
import collections
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_trec_run(run_file: str, results_dict: Dict[str, Dict[str, List[float]]], run_name: str = "QDER") -> None:
    """
    Save results in TREC run format.

    This is based on your original save_trec function from utils.py (document index 8).

    Args:
        run_file: Path to output run file
        results_dict: Dictionary with query_id -> {doc_id: [score, label]} mapping
        run_name: Name given for the run
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(run_file), exist_ok=True)

    with open(run_file, 'w') as writer:
        for query_id, doc_scores in results_dict.items():
            # Sort documents by score (descending)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)

            for rank, (doc_id, score_label) in enumerate(sorted_docs, 1):
                score = score_label[0]  # Extract score
                # TREC format: query_id Q0 doc_id rank score run_id
                writer.write(f'{query_id} Q0 {doc_id} {rank} {score} {run_name}\n')

    logger.info(f"TREC run file saved to {run_file}")


def load_trec_run(run_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load TREC run file.

    Args:
        run_file: Path to run file

    Returns:
        Dictionary with query_id -> {doc_id: score} mapping
    """
    run_data = collections.defaultdict(dict)

    with open(run_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id, _, doc_id, rank, score, run_id = parts[:6]
                    run_data[query_id][doc_id] = float(score)
                else:
                    logger.warning(f"Invalid run line {line_num}: {line.strip()}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing run line {line_num}: {e}")
                continue

    logger.info(f"Loaded run data for {len(run_data)} queries from {run_file}")
    return dict(run_data)


def load_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load TREC qrels file.

    Args:
        qrels_file: Path to qrels file

    Returns:
        Dictionary with query_id -> {doc_id: relevance} mapping
    """
    qrels = collections.defaultdict(dict)

    with open(qrels_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id, _, doc_id, relevance = parts[:4]
                    qrels[query_id][doc_id] = int(relevance)
                else:
                    logger.warning(f"Invalid qrels line {line_num}: {line.strip()}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing qrels line {line_num}: {e}")
                continue

    logger.info(f"Loaded qrels for {len(qrels)} queries from {qrels_file}")
    return dict(qrels)


def format_trec_output(query_id: str,
                       doc_id: str,
                       rank: int,
                       score: float,
                       run_id: str = 'QDER') -> str:
    """
    Format a single TREC output line.

    Args:
        query_id: Query identifier
        doc_id: Document identifier
        rank: Document rank
        score: Document score
        run_id: Run identifier

    Returns:
        Formatted TREC line
    """
    return f'{query_id} Q0 {doc_id} {rank} {score} {run_id}'


def convert_results_to_trec_format(results: Dict[str, Dict[str, float]],
                                   run_id: str = 'QDER') -> List[str]:
    """
    Convert results dictionary to list of TREC format lines.

    Args:
        results: Dictionary with query_id -> {doc_id: score} mapping
        run_id: Run identifier

    Returns:
        List of TREC format lines
    """
    trec_lines = []

    for query_id, doc_scores in results.items():
        # Sort documents by score (descending)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            line = format_trec_output(query_id, doc_id, rank, score, run_id)
            trec_lines.append(line)

    return trec_lines


def merge_run_files(run_files: List[str],
                    output_file: str,
                    merge_strategy: str = 'max') -> None:
    """
    Merge multiple run files.

    Args:
        run_files: List of run file paths
        output_file: Output merged run file
        merge_strategy: 'max', 'mean', or 'sum'
    """
    all_runs = {}

    # Load all run files
    for run_file in run_files:
        run_data = load_trec_run(run_file)
        for query_id, doc_scores in run_data.items():
            if query_id not in all_runs:
                all_runs[query_id] = collections.defaultdict(list)

            for doc_id, score in doc_scores.items():
                all_runs[query_id][doc_id].append(score)

    # Merge scores based on strategy
    merged_results = {}
    for query_id, doc_score_lists in all_runs.items():
        merged_results[query_id] = {}

        for doc_id, scores in doc_score_lists.items():
            if merge_strategy == 'max':
                merged_score = max(scores)
            elif merge_strategy == 'mean':
                merged_score = sum(scores) / len(scores)
            elif merge_strategy == 'sum':
                merged_score = sum(scores)
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")

            merged_results[query_id][doc_id] = merged_score

    # Convert to TREC format and save
    trec_lines = convert_results_to_trec_format(merged_results)

    with open(output_file, 'w') as f:
        for line in trec_lines:
            f.write(line + '\n')

    logger.info(f"Merged {len(run_files)} run files to {output_file}")


def filter_run_by_queries(run_file: str,
                          query_ids: List[str],
                          output_file: str) -> None:
    """
    Filter run file to include only specific queries.

    Args:
        run_file: Input run file
        query_ids: List of query IDs to keep
        output_file: Output filtered run file
    """
    query_set = set(query_ids)

    with open(run_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            query_id = line.strip().split()[0]
            if query_id in query_set:
                f_out.write(line)

    logger.info(f"Filtered run file saved to {output_file}")


def get_run_statistics(run_file: str) -> Dict[str, Any]:
    """
    Get statistics about a run file.

    Args:
        run_file: Path to run file

    Returns:
        Dictionary with run statistics
    """
    run_data = load_trec_run(run_file)

    total_queries = len(run_data)
    total_docs = sum(len(docs) for docs in run_data.values())

    docs_per_query = [len(docs) for docs in run_data.values()]
    avg_docs_per_query = sum(docs_per_query) / len(docs_per_query) if docs_per_query else 0

    all_scores = []
    for docs in run_data.values():
        all_scores.extend(docs.values())

    stats = {
        'total_queries': total_queries,
        'total_documents': total_docs,
        'avg_docs_per_query': avg_docs_per_query,
        'min_docs_per_query': min(docs_per_query) if docs_per_query else 0,
        'max_docs_per_query': max(docs_per_query) if docs_per_query else 0,
        'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'min_score': min(all_scores) if all_scores else 0,
        'max_score': max(all_scores) if all_scores else 0
    }

    return stats


def validate_run_file(run_file: str,
                      qrels_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate a run file format and content.

    Args:
        run_file: Path to run file
        qrels_file: Optional path to qrels file for validation

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid_format': True,
        'errors': [],
        'warnings': []
    }

    try:
        # Check basic format
        run_data = load_trec_run(run_file)

        if not run_data:
            validation_results['valid_format'] = False
            validation_results['errors'].append("No valid data found in run file")
            return validation_results

        # Check against qrels if provided
        if qrels_file:
            qrels_data = load_qrels(qrels_file)
            qrels_queries = set(qrels_data.keys())
            run_queries = set(run_data.keys())

            missing_queries = qrels_queries - run_queries
            extra_queries = run_queries - qrels_queries

            if missing_queries:
                validation_results['warnings'].append(
                    f"Run missing {len(missing_queries)} queries from qrels"
                )

            if extra_queries:
                validation_results['warnings'].append(
                    f"Run has {len(extra_queries)} queries not in qrels"
                )

        # Get basic statistics
        stats = get_run_statistics(run_file)
        validation_results['statistics'] = stats

    except Exception as e:
        validation_results['valid_format'] = False
        validation_results['errors'].append(f"Error validating run file: {e}")

    return validation_results


def create_random_baseline(qrels_file: str,
                           output_file: str,
                           docs_per_query: int = 1000) -> None:
    """
    Create a random baseline run file.

    Args:
        qrels_file: Path to qrels file
        output_file: Output run file
        docs_per_query: Number of random documents per query
    """
    import random

    # Load qrels to get query IDs
    qrels_data = load_qrels(qrels_file)

    # Get all unique document IDs
    all_docs = set()
    for docs in qrels_data.values():
        all_docs.update(docs.keys())

    all_docs = list(all_docs)

    # Generate random results
    results = {}
    for query_id in qrels_data.keys():
        # Sample random documents
        sampled_docs = random.sample(all_docs, min(docs_per_query, len(all_docs)))

        # Assign random scores
        results[query_id] = {
            doc_id: random.random() for doc_id in sampled_docs
        }

    # Save as TREC run
    trec_lines = convert_results_to_trec_format(results, 'RANDOM')

    with open(output_file, 'w') as f:
        for line in trec_lines:
            f.write(line + '\n')

    logger.info(f"Random baseline saved to {output_file}")


def truncate_run(run_file: str,
                 output_file: str,
                 max_docs_per_query: int = 1000) -> None:
    """
    Truncate run file to maximum number of documents per query.

    Args:
        run_file: Input run file
        output_file: Output truncated run file
        max_docs_per_query: Maximum documents to keep per query
    """
    run_data = load_trec_run(run_file)

    # Truncate each query's results
    truncated_results = {}
    for query_id, doc_scores in run_data.items():
        # Sort by score and take top K
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:max_docs_per_query]

        truncated_results[query_id] = dict(top_docs)

    # Save truncated run
    trec_lines = convert_results_to_trec_format(truncated_results)

    with open(output_file, 'w') as f:
        for line in trec_lines:
            f.write(line + '\n')

    logger.info(f"Truncated run file saved to {output_file}")