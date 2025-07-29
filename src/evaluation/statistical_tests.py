"""
Statistical testing utilities for CADR model evaluation.

This module provides comprehensive statistical analysis tools for comparing
information retrieval models, including significance testing, effect size
calculation, and multiple comparison correction.
"""

import os
import numpy as np
from scipy import stats
from scipy.stats import shapiro
from pytrec_eval import RelevanceEvaluator, parse_run, parse_qrel
from tabulate import tabulate
import glob
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class StatisticalTestError(Exception):
    """Custom exception for statistical testing errors."""
    pass


class TRECEvaluationError(Exception):
    """Custom exception for TREC evaluation errors."""
    pass


def read_run_file(file_path: Union[str, Path]) -> Dict:
    """
    Read and parse a TREC-format run file.

    Args:
        file_path: Path to the run file

    Returns:
        Parsed run data as dictionary

    Raises:
        TRECEvaluationError: If file cannot be read or is invalid
    """
    try:
        with open(file_path, 'r') as f:
            run_data = parse_run(f)
            if not run_data:
                raise TRECEvaluationError(f"Empty run file: {file_path}")
            return run_data
    except Exception as e:
        raise TRECEvaluationError(f"Error reading run file {file_path}: {str(e)}")


def read_qrels_file(file_path: Union[str, Path]) -> Dict:
    """
    Read and parse a TREC-format qrels file.

    Args:
        file_path: Path to the qrels file

    Returns:
        Parsed qrels data as dictionary

    Raises:
        TRECEvaluationError: If file cannot be read or is invalid
    """
    try:
        with open(file_path, 'r') as f:
            qrels_data = parse_qrel(f)
            if not qrels_data:
                raise TRECEvaluationError(f"Empty qrels file: {file_path}")
            return qrels_data
    except Exception as e:
        raise TRECEvaluationError(f"Error reading qrels file {file_path}: {str(e)}")


def calc_standard_error(differences: np.ndarray) -> float:
    """
    Calculate standard error of differences.

    Args:
        differences: Array of paired differences

    Returns:
        Standard error value
    """
    return np.std(differences, ddof=1) / np.sqrt(len(differences))


def calc_effect_size(differences: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for paired differences.

    Args:
        differences: Array of paired differences

    Returns:
        Cohen's d effect size
    """
    if np.std(differences, ddof=1) == 0:
        return 0.0
    return np.mean(differences) / np.std(differences, ddof=1)


def interpret_effect_size(effect_size: float) -> str:
    """
    Interpret Cohen's d effect size magnitude.

    Args:
        effect_size: Cohen's d value

    Returns:
        String interpretation of effect size magnitude
    """
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        return "negligible"
    elif abs_effect < 0.5:
        return "small"
    elif abs_effect < 0.8:
        return "medium"
    else:
        return "large"


def check_normality(differences: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    Perform Shapiro-Wilk test for normality.

    Args:
        differences: Array of differences to test
        alpha: Significance level for normality test

    Returns:
        Tuple of (statistic, p_value, is_normal)
    """
    if len(differences) < 3:
        logger.warning("Too few samples for normality test")
        return 0.0, 1.0, True

    statistic, p_value = shapiro(differences)
    is_normal = p_value >= alpha
    return statistic, p_value, is_normal


def perform_wilcoxon_test(reference_values: np.ndarray, test_values: np.ndarray) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test.

    Args:
        reference_values: Reference system scores
        test_values: Test system scores

    Returns:
        Tuple of (statistic, p_value)
    """
    try:
        return stats.wilcoxon(reference_values, test_values, alternative='two-sided')
    except ValueError as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return 0.0, 1.0


def perform_paired_ttest(reference_values: np.ndarray, test_values: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test.

    Args:
        reference_values: Reference system scores
        test_values: Test system scores

    Returns:
        Tuple of (statistic, p_value)
    """
    return stats.ttest_rel(reference_values, test_values)


class StatisticalTester:
    """
    Comprehensive statistical testing framework for IR evaluation.
    """

    def __init__(self, alpha: float = 0.05, correction_method: str = 'bonferroni'):
        """
        Initialize the statistical tester.

        Args:
            alpha: Significance level
            correction_method: Multiple comparison correction method
        """
        self.alpha = alpha
        self.correction_method = correction_method

    def compare_single_pair(
            self,
            reference_run: Dict,
            test_run: Dict,
            qrels: Dict,
            measure: str,
            rel_level: Optional[int] = None
    ) -> Dict:
        """
        Compare two systems using statistical tests.

        Args:
            reference_run: Reference system run data
            test_run: Test system run data
            qrels: Relevance judgments
            measure: Evaluation measure to use
            rel_level: Minimum relevance level

        Returns:
            Dictionary containing test results
        """
        # Evaluate both runs
        if rel_level:
            evaluator = RelevanceEvaluator(qrels, {measure}, relevance_level=rel_level)
        else:
            evaluator = RelevanceEvaluator(qrels, {measure})

        reference_results = evaluator.evaluate(reference_run)
        test_results = evaluator.evaluate(test_run)

        # Find common queries
        common_queries = sorted(set(reference_results.keys()) & set(test_results.keys()))

        if len(common_queries) == 0:
            raise StatisticalTestError("No common queries found between runs")

        # Extract scores
        reference_scores = np.array([reference_results[qid][measure] for qid in common_queries])
        test_scores = np.array([test_results[qid][measure] for qid in common_queries])
        differences = reference_scores - test_scores

        # Check normality
        norm_stat, norm_p, is_normal = check_normality(differences, self.alpha)

        # Choose appropriate test
        if is_normal and len(differences) >= 30:
            test_stat, p_value = perform_paired_ttest(reference_scores, test_scores)
            test_type = "paired_t_test"
        else:
            test_stat, p_value = perform_wilcoxon_test(reference_scores, test_scores)
            test_type = "wilcoxon_signed_rank"

        # Calculate additional statistics
        standard_error = calc_standard_error(differences)
        effect_size = calc_effect_size(differences)
        effect_interpretation = interpret_effect_size(effect_size)

        # Calculate descriptive statistics
        ref_mean = np.mean(reference_scores)
        test_mean = np.mean(test_scores)
        mean_diff = ref_mean - test_mean

        return {
            'test_type': test_type,
            'test_statistic': float(test_stat),
            'p_value': float(p_value),
            'is_significant': p_value < self.alpha,
            'effect_size': float(effect_size),
            'effect_interpretation': effect_interpretation,
            'standard_error': float(standard_error),
            'sample_size': len(common_queries),
            'reference_mean': float(ref_mean),
            'test_mean': float(test_mean),
            'mean_difference': float(mean_diff),
            'normality_test': {
                'statistic': float(norm_stat),
                'p_value': float(norm_p),
                'is_normal': is_normal
            },
            'common_queries': common_queries
        }

    def compare_multiple_systems(
            self,
            reference_run_file: Union[str, Path],
            test_run_files: List[Union[str, Path]],
            qrels_file: Union[str, Path],
            measure: str,
            rel_level: Optional[int] = None
    ) -> Dict:
        """
        Compare reference system against multiple test systems.

        Args:
            reference_run_file: Path to reference run file
            test_run_files: List of paths to test run files
            qrels_file: Path to qrels file
            measure: Evaluation measure
            rel_level: Minimum relevance level

        Returns:
            Dictionary containing comparison results
        """
        # Load reference run and qrels
        reference_run = read_run_file(reference_run_file)
        qrels = read_qrels_file(qrels_file)

        results = []
        p_values = []

        for test_file in test_run_files:
            try:
                test_run = read_run_file(test_file)

                # Compare systems
                comparison_result = self.compare_single_pair(
                    reference_run, test_run, qrels, measure, rel_level
                )

                comparison_result['test_file'] = str(test_file)
                comparison_result['test_filename'] = Path(test_file).name

                results.append(comparison_result)
                p_values.append(comparison_result['p_value'])

            except Exception as e:
                logger.error(f"Error processing {test_file}: {e}")
                continue

        if not results:
            raise StatisticalTestError("No valid comparisons could be performed")

        # Apply multiple comparison correction
        if len(p_values) > 1:
            _, p_values_corrected, _, _ = multipletests(
                p_values, method=self.correction_method, alpha=self.alpha
            )

            for i, result in enumerate(results):
                result['p_value_corrected'] = float(p_values_corrected[i])
                result['is_significant_corrected'] = p_values_corrected[i] < self.alpha
        else:
            for result in results:
                result['p_value_corrected'] = result['p_value']
                result['is_significant_corrected'] = result['is_significant']

        return {
            'reference_file': str(reference_run_file),
            'measure': measure,
            'rel_level': rel_level,
            'alpha': self.alpha,
            'correction_method': self.correction_method,
            'num_comparisons': len(results),
            'results': results
        }

    def compare_from_directory(
            self,
            reference_run_file: Union[str, Path],
            test_runs_directory: Union[str, Path],
            qrels_file: Union[str, Path],
            measure: str,
            rel_level: Optional[int] = None,
            run_pattern: str = "*.run*"
    ) -> Dict:
        """
        Compare reference system against all run files in a directory.

        Args:
            reference_run_file: Path to reference run file
            test_runs_directory: Directory containing test run files
            qrels_file: Path to qrels file
            measure: Evaluation measure
            rel_level: Minimum relevance level
            run_pattern: File pattern to match run files

        Returns:
            Dictionary containing comparison results
        """
        test_run_files = list(Path(test_runs_directory).glob(run_pattern))

        if not test_run_files:
            raise StatisticalTestError(
                f"No run files found in {test_runs_directory} matching pattern {run_pattern}"
            )

        return self.compare_multiple_systems(
            reference_run_file, test_run_files, qrels_file, measure, rel_level
        )


class ResultsFormatter:
    """
    Format and display statistical test results.
    """

    @staticmethod
    def format_table(comparison_results: Dict, sort_by: str = 'p_value_corrected') -> str:
        """
        Format comparison results as a table.

        Args:
            comparison_results: Results from StatisticalTester
            sort_by: Column to sort by

        Returns:
            Formatted table string
        """
        results = comparison_results['results']

        # Sort results
        if sort_by in results[0]:
            results = sorted(results, key=lambda x: x[sort_by])

        # Prepare table data
        table_data = []
        for result in results:
            table_data.append([
                result['test_filename'],
                f"{result['test_statistic']:.4f}",
                f"{result['p_value']:.4f}",
                f"{result['p_value_corrected']:.4f}",
                f"{result['standard_error']:.4f}",
                f"{result['effect_size']:.4f}",
                result['effect_interpretation'],
                result['sample_size'],
                "✓" if result['is_significant_corrected'] else "✗"
            ])

        headers = [
            "Test File", "Test Statistic", "P-Value", "P-Value (corrected)",
            "Standard Error", "Effect Size", "Effect Magnitude", "Sample Size", "Significant"
        ]

        return tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f")

    @staticmethod
    def format_summary(comparison_results: Dict) -> str:
        """
        Format a summary of the comparison results.

        Args:
            comparison_results: Results from StatisticalTester

        Returns:
            Formatted summary string
        """
        results = comparison_results['results']

        summary_lines = [
            f"Statistical Comparison Summary",
            f"=" * 40,
            f"Reference file: {comparison_results['reference_file']}",
            f"Measure: {comparison_results['measure']}",
            f"Significance level: {comparison_results['alpha']}",
            f"Correction method: {comparison_results['correction_method']}",
            f"Number of comparisons: {comparison_results['num_comparisons']}",
            "",
            f"Significant improvements: {sum(1 for r in results if r['is_significant_corrected'] and r['mean_difference'] > 0)}",
            f"Significant degradations: {sum(1 for r in results if r['is_significant_corrected'] and r['mean_difference'] < 0)}",
            f"Non-significant differences: {sum(1 for r in results if not r['is_significant_corrected'])}",
            ""
        ]

        # Add effect size summary
        large_effects = sum(1 for r in results if r['effect_interpretation'] == 'large')
        medium_effects = sum(1 for r in results if r['effect_interpretation'] == 'medium')
        small_effects = sum(1 for r in results if r['effect_interpretation'] == 'small')

        summary_lines.extend([
            f"Effect sizes:",
            f"  Large effects: {large_effects}",
            f"  Medium effects: {medium_effects}",
            f"  Small effects: {small_effects}",
        ])

        return "\n".join(summary_lines)

    @staticmethod
    def save_results(comparison_results: Dict, output_file: Union[str, Path]) -> None:
        """
        Save comparison results to a file.

        Args:
            comparison_results: Results from StatisticalTester
            output_file: Path to output file
        """
        output_path = Path(output_file)

        with open(output_path, 'w') as f:
            f.write(ResultsFormatter.format_summary(comparison_results))
            f.write("\n\n")
            f.write(ResultsFormatter.format_table(comparison_results))

        logger.info(f"Results saved to {output_path}")


def paired_t_test(
        reference_run_file: Union[str, Path],
        test_run_files_dir: Union[str, Path],
        qrels_file: Union[str, Path],
        eval_measure: str,
        rel_level: Optional[int] = None,
        alpha: float = 0.05,
        correction_method: str = 'bonferroni',
        output_file: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Convenience function for paired t-test analysis.

    Args:
        reference_run_file: Path to reference run file
        test_run_files_dir: Directory containing test run files
        qrels_file: Path to qrels file
        eval_measure: Evaluation measure to use
        rel_level: Minimum relevance level for judgments
        alpha: Significance level
        correction_method: Multiple comparison correction method
        output_file: Optional output file to save results

    Returns:
        Dictionary containing comparison results
    """
    tester = StatisticalTester(alpha=alpha, correction_method=correction_method)

    results = tester.compare_from_directory(
        reference_run_file=reference_run_file,
        test_runs_directory=test_run_files_dir,
        qrels_file=qrels_file,
        measure=eval_measure,
        rel_level=rel_level
    )

    # Print results
    print(ResultsFormatter.format_summary(results))
    print("\n")
    print(ResultsFormatter.format_table(results))

    # Save if requested
    if output_file:
        ResultsFormatter.save_results(results, output_file)

    return results