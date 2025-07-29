"""
Data loading and file I/O utilities for QDER.
"""

import json
import collections
from typing import Dict, List, Optional, Any
from tqdm import tqdm


def load_entity_embeddings(embedding_file: str) -> tuple:
    """
    Load entity embeddings with names for alignment.

    Args:
        embedding_file: Path to entity embeddings file (gzipped JSONL)

    Returns:
        Tuple of (embeddings_dict, entity_names_dict)
    """
    import gzip

    embeddings = {}
    entity_names = {}

    print("Loading entity embeddings with names...")

    open_func = gzip.open if embedding_file.endswith('.gz') else open
    mode = 'rt' if embedding_file.endswith('.gz') else 'r'

    with open_func(embedding_file, mode, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            try:
                data = json.loads(line.strip())
                entity_id = data['entity_id']
                embeddings[entity_id] = data['embedding']
                entity_names[entity_id] = data.get('entity_name', entity_id)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid embedding line: {e}")
                continue

    print(f"Loaded {len(embeddings)} entity embeddings")
    return embeddings, entity_names


def load_queries(queries_file: str) -> Dict[str, str]:
    """
    Load queries from TSV file.

    Args:
        queries_file: Path to queries file (TSV format: query_id\\tquery_text)

    Returns:
        Dictionary mapping query_id -> query_text
    """
    queries = {}

    with open(queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\\t')
                if len(parts) >= 2:
                    query_id, query_text = parts[0], parts[1]
                    queries[query_id] = query_text
                else:
                    print(f"Skipping invalid query line {line_num}: {line.strip()}")
            except Exception as e:
                print(f"Error processing query line {line_num}: {e}")
                continue

    print(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load TREC qrels file.

    Args:
        qrels_file: Path to qrels file (TREC format)

    Returns:
        Dictionary mapping query_id -> {doc_id: relevance}
    """
    qrels = collections.defaultdict(dict)

    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id, _, doc_id, relevance = parts[0], parts[1], parts[2], int(parts[3])
                    qrels[query_id][doc_id] = relevance
                else:
                    print(f"Skipping invalid qrels line {line_num}: {line.strip()}")
            except (ValueError, IndexError) as e:
                print(f"Error processing qrels line {line_num}: {e}")
                continue

    print(f"Loaded qrels for {len(qrels)} queries")
    return dict(qrels)


def load_run_file(run_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load TREC run file.

    Args:
        run_file: Path to run file (TREC format)

    Returns:
        Dictionary mapping query_id -> {doc_id: score}
    """
    run_data = collections.defaultdict(dict)

    with open(run_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id, _, doc_id, rank, score, run_id = parts[:6]
                    run_data[query_id][doc_id] = float(score)
                else:
                    print(f"Skipping invalid run line {line_num}: {line.strip()}")
            except (ValueError, IndexError) as e:
                print(f"Error processing run line {line_num}: {e}")
                continue

    print(f"Loaded run data for {len(run_data)} queries")
    return dict(run_data)


def load_docs_from_jsonl(docs_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load documents from JSONL file.

    Args:
        docs_file: Path to documents JSONL file

    Returns:
        Dictionary mapping doc_id -> document_data
    """
    docs = {}

    print(f"Loading documents from {docs_file}...")
    with open(docs_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading documents"), 1):
            try:
                doc = json.loads(line.strip())
                if 'doc_id' in doc:
                    docs[doc['doc_id']] = doc
                else:
                    print(f"Document missing doc_id on line {line_num}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {line_num}: {e}")
                continue

    print(f"Loaded {len(docs)} documents")
    return docs


def save_jsonl(data: List[Dict], output_file: str) -> None:
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries to save
        output_file: Path to output file
    """
    print(f"Saving {len(data)} items to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\\n')
    print("Save completed")


def validate_dataset_file(dataset_file: str, required_fields: Optional[List[str]] = None) -> bool:
    """
    Validate that a dataset file has the required format and fields.

    Args:
        dataset_file: Path to dataset JSONL file
        required_fields: List of required field names

    Returns:
        True if valid, False otherwise
    """
    if required_fields is None:
        required_fields = ['query', 'doc', 'label']

    print(f"Validating dataset file: {dataset_file}")

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            # Check first few lines
            for i, line in enumerate(f):
                if i >= 10:  # Check first 10 lines
                    break

                try:
                    data = json.loads(line.strip())

                    # Check required fields
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        print(f"Line {i + 1} missing required fields: {missing_fields}")
                        return False

                except json.JSONDecodeError as e:
                    print(f"Invalid JSON on line {i + 1}: {e}")
                    return False

        print("Dataset file validation passed")
        return True

    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_file}")
        return False
    except Exception as e:
        print(f"Error validating dataset file: {e}")
        return False


def create_balanced_dataset(input_file: str, output_file: str, max_examples_per_label: int = None) -> None:
    """
    Create a balanced dataset by sampling equal numbers of positive and negative examples.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output balanced JSONL file
        max_examples_per_label: Maximum examples per label (None for no limit)
    """
    positive_examples = []
    negative_examples = []

    # Load and separate examples by label
    print(f"Loading examples from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing examples"):
            try:
                example = json.loads(line.strip())
                if example.get('label') == 1:
                    positive_examples.append(example)
                else:
                    negative_examples.append(example)
            except json.JSONDecodeError:
                continue

    print(f"Found {len(positive_examples)} positive and {len(negative_examples)} negative examples")

    # Balance the dataset
    min_count = min(len(positive_examples), len(negative_examples))
    if max_examples_per_label:
        min_count = min(min_count, max_examples_per_label)

    # Sample examples
    import random
    random.shuffle(positive_examples)
    random.shuffle(negative_examples)

    balanced_examples = positive_examples[:min_count] + negative_examples[:min_count]
    random.shuffle(balanced_examples)

    # Save balanced dataset
    save_jsonl(balanced_examples, output_file)
    print(f"Created balanced dataset with {min_count} examples per label")


def count_dataset_stats(dataset_file: str) -> Dict[str, Any]:
    """
    Count statistics for a dataset file.

    Args:
        dataset_file: Path to dataset JSONL file

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_examples': 0,
        'positive_examples': 0,
        'negative_examples': 0,
        'has_entities': 0,
        'avg_query_length': 0,
        'avg_doc_length': 0
    }

    query_lengths = []
    doc_lengths = []

    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Analyzing dataset"):
            try:
                example = json.loads(line.strip())
                stats['total_examples'] += 1

                if example.get('label') == 1:
                    stats['positive_examples'] += 1
                else:
                    stats['negative_examples'] += 1

                if example.get('query_ent_emb') or example.get('doc_ent_emb'):
                    stats['has_entities'] += 1

                if 'query' in example:
                    query_lengths.append(len(example['query'].split()))

                if 'doc' in example:
                    doc_lengths.append(len(example['doc'].split()))

            except json.JSONDecodeError:
                continue

    if query_lengths:
        stats['avg_query_length'] = sum(query_lengths) / len(query_lengths)
    if doc_lengths:
        stats['avg_doc_length'] = sum(doc_lengths) / len(doc_lengths)

    return stats