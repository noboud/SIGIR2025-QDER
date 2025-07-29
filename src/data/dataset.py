"""
Dataset classes for QDER model training and evaluation.
"""

from typing import Dict, Any, Optional, List
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class QDERDataset(Dataset):
    """
    Dataset for QDER (Query-Document Entity Ranking) model training and evaluation.

    Loads query-document pairs with entity embeddings and relevance labels.
    Handles both training and evaluation modes with different return formats.
    """

    def __init__(self,
                 dataset: str,
                 tokenizer,
                 train: bool,
                 max_len: int):
        """
        Initialize the dataset.

        Args:
            dataset: Path to JSONL dataset file
            tokenizer: Tokenizer for text processing
            train: Whether this is training data (affects return format)
            max_len: Maximum sequence length for tokenization
        """
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._train = train

        # Load data
        self._read_data()
        self._count = len(self._examples)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return self._count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.

        Args:
            idx: Index of the example to retrieve

        Returns:
            Dictionary containing processed example data
        """
        example: Dict[str, Any] = self._examples[idx]

        # Create tokenized inputs for query and document
        query_input_ids, query_attention_mask, query_token_type_ids = self._create_input(
            text=example['query']
        )
        doc_input_ids, doc_attention_mask, doc_token_type_ids = self._create_input(
            text=example['doc']
        )

        # Get entity embeddings
        doc_entity_emb = example.get('doc_ent_emb', [])
        query_entity_emb = example.get('query_ent_emb', [])
        doc_score = example.get('doc_score', 1.0)

        # Return different formats for training vs evaluation
        if self._train:
            return {
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'query_entity_emb': query_entity_emb,
                'doc_input_ids': doc_input_ids,
                'doc_attention_mask': doc_attention_mask,
                'doc_token_type_ids': doc_token_type_ids,
                'doc_entity_emb': doc_entity_emb,
                'doc_score': doc_score,
                'label': example['label']
            }
        else:
            return {
                'query_id': example.get('query_id', ''),
                'doc_id': example.get('doc_id', ''),
                'label': example['label'],
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'query_entity_emb': query_entity_emb,
                'doc_input_ids': doc_input_ids,
                'doc_attention_mask': doc_attention_mask,
                'doc_token_type_ids': doc_token_type_ids,
                'doc_entity_emb': doc_entity_emb,
                'doc_score': doc_score,
            }

    def _create_input(self, text: str) -> tuple:
        """
        Tokenize and encode input text.

        Args:
            text: Input text to tokenize

        Returns:
            Tuple of (input_ids, attention_mask, token_type_ids)
        """
        encoded_dict = self._tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_len,  # Pad & truncate all sentences
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attention masks
            return_token_type_ids=True  # Construct token type ids
        )

        return (
            encoded_dict['input_ids'],
            encoded_dict['attention_mask'],
            encoded_dict['token_type_ids']
        )

    def _read_data(self) -> None:
        """Load examples from the dataset file."""
        print(f"Loading dataset from {self._dataset}...")
        self._examples = []

        with open(self._dataset, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading examples"):
                try:
                    example = json.loads(line.strip())

                    # Validate required fields
                    if 'query' not in example or 'doc' not in example or 'label' not in example:
                        continue

                    # Ensure entity embeddings are lists
                    if 'query_ent_emb' in example and not isinstance(example['query_ent_emb'], list):
                        example['query_ent_emb'] = []
                    if 'doc_ent_emb' in example and not isinstance(example['doc_ent_emb'], list):
                        example['doc_ent_emb'] = []

                    # Set default values for optional fields
                    if 'query_ent_emb' not in example:
                        example['query_ent_emb'] = []
                    if 'doc_ent_emb' not in example:
                        example['doc_ent_emb'] = []
                    if 'doc_score' not in example:
                        example['doc_score'] = 1.0

                    self._examples.append(example)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping invalid line: {e}")
                    continue

        print(f"Loaded {len(self._examples)} examples")

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched tensors ready for model input
        """
        # Convert lists to tensors
        query_input_ids = torch.tensor([item['query_input_ids'] for item in batch])
        query_attention_mask = torch.tensor([item['query_attention_mask'] for item in batch])
        query_token_type_ids = torch.tensor([item['query_token_type_ids'] for item in batch])
        doc_input_ids = torch.tensor([item['doc_input_ids'] for item in batch])
        doc_attention_mask = torch.tensor([item['doc_attention_mask'] for item in batch])
        doc_token_type_ids = torch.tensor([item['doc_token_type_ids'] for item in batch])
        doc_score = torch.tensor([item['doc_score'] for item in batch], dtype=torch.float)

        # Handle labels
        label = torch.tensor([item['label'] for item in batch], dtype=torch.float)

        # Handle entity embeddings - pad sequences since they can have different lengths
        query_entity_emb_list = [torch.tensor(item['query_entity_emb']) if item['query_entity_emb'] else torch.empty(0, 300) for item in batch]
        doc_entity_emb_list = [torch.tensor(item['doc_entity_emb']) if item['doc_entity_emb'] else torch.empty(0, 300) for item in batch]

        # Pad entity embeddings
        if all(len(emb) > 0 for emb in query_entity_emb_list):
            query_entity_emb = pad_sequence(query_entity_emb_list, batch_first=True)
        else:
            # Handle case where some examples have no entities
            max_len = max(len(emb) for emb in query_entity_emb_list) if query_entity_emb_list else 1
            query_entity_emb = torch.zeros(len(batch), max_len, 300)
            for i, emb in enumerate(query_entity_emb_list):
                if len(emb) > 0:
                    query_entity_emb[i, :len(emb)] = emb

        if all(len(emb) > 0 for emb in doc_entity_emb_list):
            doc_entity_emb = pad_sequence(doc_entity_emb_list, batch_first=True)
        else:
            # Handle case where some examples have no entities
            max_len = max(len(emb) for emb in doc_entity_emb_list) if doc_entity_emb_list else 1
            doc_entity_emb = torch.zeros(len(batch), max_len, 300)
            for i, emb in enumerate(doc_entity_emb_list):
                if len(emb) > 0:
                    doc_entity_emb[i, :len(emb)] = emb

        # Create entity attention masks
        query_entity_lengths = [len(item['query_entity_emb']) if item['query_entity_emb'] else 0 for item in batch]
        doc_entity_lengths = [len(item['doc_entity_emb']) if item['doc_entity_emb'] else 0 for item in batch]

        max_query_entity_len = query_entity_emb.size(1)
        max_doc_entity_len = doc_entity_emb.size(1)

        query_entity_mask = torch.zeros(len(batch), max_query_entity_len, dtype=torch.bool)
        doc_entity_mask = torch.zeros(len(batch), max_doc_entity_len, dtype=torch.bool)

        for i, (q_len, d_len) in enumerate(zip(query_entity_lengths, doc_entity_lengths)):
            if q_len > 0:
                query_entity_mask[i, :q_len] = True
            if d_len > 0:
                doc_entity_mask[i, :d_len] = True

        # Prepare return dictionary based on training vs evaluation mode
        result = {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'query_token_type_ids': query_token_type_ids,
            'query_entity_emb': query_entity_emb,
            'query_entity_mask': query_entity_mask,
            'doc_input_ids': doc_input_ids,
            'doc_attention_mask': doc_attention_mask,
            'doc_token_type_ids': doc_token_type_ids,
            'doc_entity_emb': doc_entity_emb,
            'doc_entity_mask': doc_entity_mask,
            'doc_score': doc_score,
            'label': label
        }

        # Add query_id and doc_id for evaluation mode
        if not self._train:
            result['query_id'] = [item['query_id'] for item in batch]
            result['doc_id'] = [item['doc_id'] for item in batch]

        return result