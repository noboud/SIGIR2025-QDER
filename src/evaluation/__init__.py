"""
Evaluation components for QDER models.

This package contains:
- Model evaluation functionality
- TREC evaluation metrics using pytrec_eval
- Ranking utilities and run file processing
- Statistical significance testing
"""

from .evaluator import QDERModelEvaluator
from .metrics import get_metric, evaluate_run, compute_ranking_metrics
from .ranking_utils import save_trec_run, load_trec_run, load_qrels, format_trec_output
from .statistical_tests import (
    paired_t_test,
    wilcoxon_test,
    compute_effect_size,
    multiple_comparison_correction
)

__all__ = [
    'QDERModelEvaluator',
    'get_metric',
    'evaluate_run',
    'compute_ranking_metrics',
    'save_trec_run',
    'load_trec_run',
    'load_qrels',
    'format_trec_output',
    'paired_t_test',
    'wilcoxon_test',
    'compute_effect_size',
    'multiple_comparison_correction'
]