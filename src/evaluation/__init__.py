"""
Evaluation components for CADR models.

This package contains:
- Model evaluation functionality
- TREC evaluation metrics using pytrec_eval
- Ranking utilities and run file processing
- Comprehensive statistical significance testing framework
- Effect size calculation and interpretation
- Multiple comparison correction methods
"""

from .evaluator import QDERModelEvaluator
from .metrics import get_metric, evaluate_run, compute_ranking_metrics
from .ranking_utils import save_trec_run, load_trec_run, load_qrels, format_trec_output
from .statistical_tests import (
    # Main convenience function (backward compatibility)
    paired_t_test,

    # Core classes
    StatisticalTester,
    ResultsFormatter,

    # Utility functions
    read_run_file,
    read_qrels_file,
    calc_standard_error,
    calc_effect_size,
    interpret_effect_size,
    check_normality,
    perform_wilcoxon_test,
    perform_paired_ttest,

    # Custom exceptions
    StatisticalTestError,
    TRECEvaluationError,
)

# Legacy imports for backward compatibility
# These can be deprecated in future versions
def wilcoxon_test(*args, **kwargs):
    """Deprecated: Use StatisticalTester.compare_single_pair() instead."""
    import warnings
    warnings.warn(
        "wilcoxon_test is deprecated. Use StatisticalTester.compare_single_pair() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return perform_wilcoxon_test(*args, **kwargs)

def compute_effect_size(*args, **kwargs):
    """Deprecated: Use calc_effect_size() instead."""
    import warnings
    warnings.warn(
        "compute_effect_size is deprecated. Use calc_effect_size() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return calc_effect_size(*args, **kwargs)

def multiple_comparison_correction(*args, **kwargs):
    """Deprecated: Use StatisticalTester with correction_method parameter instead."""
    import warnings
    warnings.warn(
        "multiple_comparison_correction is deprecated. Use StatisticalTester instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from statsmodels.stats.multitest import multipletests
    return multipletests(*args, **kwargs)

__all__ = [
    # Core evaluation
    'QDERModelEvaluator',
    'get_metric',
    'evaluate_run',
    'compute_ranking_metrics',

    # Ranking utilities
    'save_trec_run',
    'load_trec_run',
    'load_qrels',
    'format_trec_output',

    # Statistical testing - main API
    'paired_t_test',
    'StatisticalTester',
    'ResultsFormatter',

    # Statistical testing - utility functions
    'read_run_file',
    'read_qrels_file',
    'calc_standard_error',
    'calc_effect_size',
    'interpret_effect_size',
    'check_normality',
    'perform_wilcoxon_test',
    'perform_paired_ttest',

    # Exceptions
    'StatisticalTestError',
    'TRECEvaluationError',

    # Legacy functions (deprecated)
    'wilcoxon_test',
    'compute_effect_size',
    'multiple_comparison_correction',
]

# Version info
__version__ = '1.0.0'
__author__ = 'CADR Research Team'