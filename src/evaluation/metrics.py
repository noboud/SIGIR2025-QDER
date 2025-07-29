"""
Evaluation metrics using pytrec_eval.

This contains your original metrics.py code (document index 4) integrated
into the modular structure.
"""

import pytrec_eval
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def get_metric(qrels: str, run: str, metric: str = 'map') -> float:
    """
    Compute a single evaluation metric using pytrec_eval.

    This is your original function from document index 4.

    Args:
        qrels: Path to qrels file in TREC format
        run: Path to run file in TREC format
        metric: Metric to compute (default: 'map')

    Returns:
        Computed metric value
    """
    # Read the qrel file
    with open(qrels, 'r') as f_qrel:
        qrel_dict = pytrec_eval.parse_qrel(f_qrel)

    # Read the run file
    with open(run, 'r') as f_run:
        run_dict = pytrec_eval.parse_run(f_run)

    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run_dict)

    # Aggregate results
    mes = {}
    for _, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            mes[measure] = pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()]
            )

    return mes[metric]


def evaluate_run(qrels_file: str,
                 run_file: str,
                 metric: str = 'map',
                 relevance_level: Optional[int] = None) -> float:
    """
    Evaluate a run file against qrels.

    Args:
        qrels_file: Path to qrels file
        run_file: Path to run file
        metric: Metric to compute
        relevance_level: Minimum relevance level (None for default)

    Returns:
        Computed metric value
    """
    try:
        # Read qrels and run files
        with open(qrels_file, 'r') as f:
            qrels = pytrec_eval.parse_qrel(f)

        with open(run_file, 'r') as f:
            run = pytrec_eval.parse_run(f)

        # Create evaluator
        if relevance_level is not None:
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels,
                {metric},
                relevance_level=relevance_level
            )
        else:
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric})

        # Evaluate
        results = evaluator.evaluate(run)

        # Compute aggregated metric
        metric_values = [measures[metric] for measures in results.values()]
        aggregated_value = pytrec_eval.compute_aggregated_measure(metric, metric_values)

        return aggregated_value

    except Exception as e:
        logger.error(f"Error evaluating run: {e}")
        raise


def compute_ranking_metrics(qrels_file: str,
                            run_file: str,
                            metrics: Optional[List[str]] = None,
                            relevance_level: Optional[int] = None) -> Dict[str, float]:
    """
    Compute multiple ranking metrics.

    Args:
        qrels_file: Path to qrels file
        run_file: Path to run file
        metrics: List of metrics to compute (None for standard set)
        relevance_level: Minimum relevance level

    Returns:
        Dictionary with computed metrics
    """
    if metrics is None:
        metrics = [
            'map', 'gm_map',  # Mean Average Precision
            'ndcg', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20',  # NDCG
            'P_5', 'P_10', 'P_20',  # Precision at k
            'recall_5', 'recall_10', 'recall_20',  # Recall at k
            'recip_rank',  # Mean Reciprocal Rank
            'bpref'  # Binary Preference
        ]

    try:
        # Read files
        with open(qrels_file, 'r') as f:
            qrels = pytrec_eval.parse_qrel(f)

        with open(run_file, 'r') as f:
            run = pytrec_eval.parse_run(f)

        # Create evaluator with all metrics
        if relevance_level is not None:
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels,
                metrics,
                relevance_level=relevance_level
            )
        else:
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

        # Evaluate
        results = evaluator.evaluate(run)

        # Compute aggregated metrics
        aggregated_metrics = {}
        for metric in metrics:
            try:
                metric_values = [measures[metric] for measures in results.values() if metric in measures]
                if metric_values:
                    aggregated_metrics[metric] = pytrec_eval.compute_aggregated_measure(metric, metric_values)
                else:
                    logger.warning(f"No values found for metric: {metric}")
                    aggregated_metrics[metric] = 0.0
            except Exception as e:
                logger.warning(f"Error computing metric {metric}: {e}")
                aggregated_metrics[metric] = 0.0

        return aggregated_metrics

    except Exception as e:
        logger.error(f"Error computing ranking metrics: {e}")
        raise


def compute_per_query_metrics(qrels_file: str,
                              run_file: str,
                              metrics: Optional[List[str]] = None,
                              relevance_level: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute per-query metrics (not aggregated).

    Args:
        qrels_file: Path to qrels file
        run_file: Path to run file
        metrics: List of metrics to compute
        relevance_level: Minimum relevance level

    Returns:
        Dictionary with per-query metrics {query_id: {metric: value}}
    """
    if metrics is None:
        metrics = ['map', 'ndcg', 'P_10', 'recip_rank']

    try:
        # Read files
        with open(qrels_file, 'r') as f:
            qrels = pytrec_eval.parse_qrel(f)

        with open(run_file, 'r') as f:
            run = pytrec_eval.parse_run(f)

        # Create evaluator
        if relevance_level is not None:
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels,
                metrics,
                relevance_level=relevance_level
            )
        else:
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

        # Evaluate and return per-query results
        results = evaluator.evaluate(run)

        return results

    except Exception as e:
        logger.error(f"Error computing per-query metrics: {e}")
        raise


def compare_runs(qrels_file: str,
                 run_file_1: str,
                 run_file_2: str,
                 metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare two run files.

    Args:
        qrels_file: Path to qrels file
        run_file_1: Path to first run file
        run_file_2: Path to second run file
        metrics: List of metrics to compute

    Returns:
        Dictionary with comparison results
    """
    if metrics is None:
        metrics = ['map', 'ndcg', 'P_10', 'recip_rank']

    # Compute metrics for both runs
    metrics_1 = compute_ranking_metrics(qrels_file, run_file_1, metrics)
    metrics_2 = compute_ranking_metrics(qrels_file, run_file_2, metrics)

    # Compute differences
    differences = {}
    for metric in metrics:
        if metric in metrics_1 and metric in metrics_2:
            differences[metric] = metrics_2[metric] - metrics_1[metric]

    return {
        'run_1': metrics_1,
        'run_2': metrics_2,
        'differences': differences,
        'improvements': {k: v > 0 for k, v in differences.items()},
        'relative_improvements': {
            k: (v / metrics_1[k] * 100) if metrics_1[k] != 0 else 0
            for k, v in differences.items()
        }
    }


def get_supported_metrics() -> List[str]:
    """
    Get list of supported pytrec_eval metrics.

    Returns:
        List of supported metric names
    """
    return list(pytrec_eval.supported_measures)


def validate_run_format(run_file: str) -> bool:
    """
    Validate that a run file is in correct TREC format.

    Args:
        run_file: Path to run file

    Returns:
        True if format is valid
    """
    try:
        with open(run_file, 'r') as f:
            pytrec_eval.parse_run(f)
        return True
    except Exception as e:
        logger.error(f"Invalid run file format: {e}")
        return False


def validate_qrels_format(qrels_file: str) -> bool:
    """
    Validate that a qrels file is in correct TREC format.

    Args:
        qrels_file: Path to qrels file

    Returns:
        True if format is valid
    """
    try:
        with open(qrels_file, 'r') as f:
            pytrec_eval.parse_qrel(f)
        return True
    except Exception as e:
        logger.error(f"Invalid qrels file format: {e}")
        return False


def filter_run_by_qrels(run_file: str,
                        qrels_file: str,
                        output_file: str) -> None:
    """
    Filter run file to only include queries that exist in qrels.

    Args:
        run_file: Path to input run file
        qrels_file: Path to qrels file
        output_file: Path to filtered output run file
    """
    # Read qrels to get valid query IDs
    with open(qrels_file, 'r') as f:
        qrels = pytrec_eval.parse_qrel(f)

    valid_queries = set(qrels.keys())

    # Filter run file
    with open(run_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            query_id = line.strip().split()[0]
            if query_id in valid_queries:
                f_out.write(line)

    logger.info(f"Filtered run file saved to {output_file}")


def get_metric_description(metric: str) -> str:
    """
    Get description of a metric.

    Args:
        metric: Metric name

    Returns:
        Description of the metric
    """
    descriptions = {
        'map': 'Mean Average Precision',
        'gm_map': 'Geometric Mean Average Precision',
        'ndcg': 'Normalized Discounted Cumulative Gain',
        'ndcg_cut_5': 'NDCG at rank 5',
        'ndcg_cut_10': 'NDCG at rank 10',
        'ndcg_cut_20': 'NDCG at rank 20',
        'P_5': 'Precision at rank 5',
        'P_10': 'Precision at rank 10',
        'P_20': 'Precision at rank 20',
        'recall_5': 'Recall at rank 5',
        'recall_10': 'Recall at rank 10',
        'recall_20': 'Recall at rank 20',
        'recip_rank': 'Mean Reciprocal Rank',
        'bpref': 'Binary Preference',
        'num_q': 'Number of queries',
        'num_ret': 'Number of retrieved documents',
        'num_rel': 'Number of relevant documents',
        'num_rel_ret': 'Number of relevant documents retrieved'
    }

    return descriptions.get(metric, f"Unknown metric: {metric}")