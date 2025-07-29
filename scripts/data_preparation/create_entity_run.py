#!/usr/bin/env python3
"""
Create entity ranking from document ranking.

This script takes a document run file and creates an entity ranking
by aggregating document scores for entities contained in those documents.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import logging
from pathlib import Path

from src.utils import setup_logging
from src.entity_processing import (
    EntityRanker,
    load_document_run,
    load_document_entities,
    write_entity_rankings_to_file,
    compute_entity_statistics
)

logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create entity ranking from document ranking"
    )

    # Required arguments
    parser.add_argument(
        "--docs",
        required=True,
        type=str,
        help="Document file with entity annotations (JSONL format)"
    )
    parser.add_argument(
        "--run",
        required=True,
        type=str,
        help="Document run file (TREC format)"
    )
    parser.add_argument(
        "--save",
        required=True,
        type=str,
        help="Output entity ranking file"
    )

    # Optional arguments
    parser.add_argument(
        "--aggregation-method",
        default="sum",
        choices=["sum", "mean", "max"],
        help="Method to aggregate entity scores (default: sum)"
    )
    parser.add_argument(
        "--normalization-method",
        default="min_max",
        choices=["min_max", "z_score", "sigmoid", "none"],
        help="Score normalization method (default: min_max)"
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Apply log transformation before normalization"
    )
    parser.add_argument(
        "--run-name",
        default="EntityRanking",
        type=str,
        help="Run name for TREC format (default: EntityRanking)"
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Save ranking statistics"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("Starting entity ranking creation")
    logger.info(f"Document file: {args.docs}")
    logger.info(f"Run file: {args.run}")
    logger.info(f"Output file: {args.save}")
    logger.info(f"Aggregation method: {args.aggregation_method}")
    logger.info(f"Normalization method: {args.normalization_method}")
    logger.info(f"Log transform: {args.log_transform}")

    try:
        # Load document run
        logger.info("Loading document run file...")
        doc_run = load_document_run(args.run)

        # Load document-entity mappings
        logger.info("Loading document-entity mappings...")
        docs = load_document_entities(args.docs)

        # Create entity ranker
        logger.info("Creating entity rankings...")
        ranker = EntityRanker(
            aggregation_method=args.aggregation_method,
            normalization_method=args.normalization_method,
            log_transform=args.log_transform
        )

        # Generate entity rankings
        entity_rankings = ranker.create_entity_rankings(doc_run, docs)

        # Write entity rankings
        logger.info("Writing entity rankings...")
        write_entity_rankings_to_file(
            entity_rankings,
            args.save,
            run_name=args.run_name
        )

        # Compute and save statistics if requested
        if args.save_stats:
            logger.info("Computing ranking statistics...")
            stats = compute_entity_statistics(entity_rankings)

            # Save statistics
            stats_file = Path(args.save).parent / f"{Path(args.save).stem}_stats.json"
            import json
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Statistics saved to: {stats_file}")

            # Print summary
            logger.info("Ranking Statistics:")
            logger.info(f"  Total queries: {stats['total_queries']}")
            logger.info(f"  Total unique entities: {stats['total_entities']}")
            logger.info(f"  Average entities per query: {stats['avg_entities_per_query']:.2f}")
            logger.info(f"  Min entities per query: {stats['min_entities_per_query']}")
            logger.info(f"  Max entities per query: {stats['max_entities_per_query']}")
            logger.info(f"  Queries with no entities: {stats['queries_with_no_entities']}")

        logger.info("Entity ranking creation completed successfully!")

    except Exception as e:
        logger.error(f"Error during entity ranking creation: {e}")
        raise


if __name__ == '__main__':
    main()