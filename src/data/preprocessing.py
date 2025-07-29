"""
Text preprocessing utilities for QDER.
"""

import re
from typing import List, Optional


def preprocess_text(text: str, lowercase: bool = True, remove_extra_spaces: bool = True) -> str:
    """
    Preprocess text for model input.

    Args:
        text: Input text to preprocess
        lowercase: Whether to convert to lowercase
        remove_extra_spaces: Whether to remove extra whitespace

    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Remove extra whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    return text


def clean_query_text(query: str) -> str:
    """
    Clean query text for better processing.

    Args:
        query: Raw query text

    Returns:
        Cleaned query text
    """
    if not query:
        return ""

    # Remove common query artifacts
    query = re.sub(r'[^\w\s]', ' ', query)  # Remove punctuation
    query = re.sub(r'\d+', ' ', query)  # Remove numbers
    query = re.sub(r'\s+', ' ', query)  # Normalize whitespace

    return query.strip()


def clean_document_text(doc_text: str, max_length: Optional[int] = None) -> str:
    """
    Clean document text for processing.

    Args:
        doc_text: Raw document text
        max_length: Optional maximum length to truncate

    Returns:
        Cleaned document text
    """
    if not doc_text:
        return ""

    # Basic cleaning
    doc_text = re.sub(r'\n+', ' ', doc_text)  # Replace newlines with spaces
    doc_text = re.sub(r'\s+', ' ', doc_text)  # Normalize whitespace
    doc_text = doc_text.strip()

    # Truncate if needed
    if max_length and len(doc_text) > max_length:
        doc_text = doc_text[:max_length].rsplit(' ', 1)[0]  # Truncate at word boundary

    return doc_text


def normalize_entity_text(entity_text: str) -> str:
    """
    Normalize entity text for consistent processing.

    Args:
        entity_text: Raw entity text/name

    Returns:
        Normalized entity text
    """
    if not entity_text:
        return ""

    # Replace underscores with spaces (common in Wikipedia entity names)
    entity_text = entity_text.replace('_', ' ')

    # Basic cleaning
    entity_text = re.sub(r'\s+', ' ', entity_text)
    entity_text = entity_text.strip()

    return entity_text


def tokenize_for_entities(text: str) -> List[str]:
    """
    Simple tokenization for entity extraction.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Simple whitespace tokenization with punctuation handling
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def prepare_text_for_model(text: str, max_length: int = 512) -> str:
    """
    Prepare text for model input with all preprocessing steps.

    Args:
        text: Raw input text
        max_length: Maximum length for the text

    Returns:
        Preprocessed text ready for tokenizer
    """
    if not text:
        return ""

    # Apply all preprocessing steps
    text = preprocess_text(text, lowercase=True, remove_extra_spaces=True)

    # Ensure reasonable length for tokenizer
    if len(text) > max_length * 4:  # Rough estimate for token count
        text = text[:max_length * 4].rsplit(' ', 1)[0]

    return text