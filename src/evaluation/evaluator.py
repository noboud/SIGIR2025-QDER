"""
Model evaluation functionality for QDER models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class QDERModelEvaluator:
    """
    Evaluator for QDER models.

    Handles model evaluation, prediction generation, and results formatting.
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda'):
        """
        Initialize evaluator.

        Args:
            model: QDER model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

    def evaluate(self, data_loader: DataLoader) -> Dict[str, Dict[str, List[float]]]:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: DataLoader with evaluation data

        Returns:
            Dictionary with query_id -> {doc_id: [score, label]} mapping
        """
        results = {}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
                batch_results = self._evaluate_batch(batch)

                # Merge batch results into overall results
                for query_id, doc_scores in batch_results.items():
                    if query_id not in results:
                        results[query_id] = {}
                    results[query_id].update(doc_scores)

        return results

    def _evaluate_batch(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """
        Evaluate a single batch.

        Args:
            batch: Batch data

        Returns:
            Dictionary with batch results
        """
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get model predictions
        outputs = self.model(
            query_input_ids=batch['query_input_ids'],
            query_attention_mask=batch['query_attention_mask'],
            query_token_type_ids=batch['query_token_type_ids'],
            query_entity_emb=batch['query_entity_emb'],
            doc_input_ids=batch['doc_input_ids'],
            doc_attention_mask=batch['doc_attention_mask'],
            doc_token_type_ids=batch['doc_token_type_ids'],
            doc_entity_emb=batch['doc_entity_emb'],
            query_entity_mask=batch.get('query_entity_mask'),
            doc_entity_mask=batch.get('doc_entity_mask'),
            doc_scores=batch.get('doc_score')
        )

        # Extract predictions
        scores = outputs['score'].detach().cpu().tolist()
        query_ids = batch['query_id']
        doc_ids = batch['doc_id']
        labels = batch['label'].cpu().tolist()

        # Organize results by query
        batch_results = {}
        for query_id, doc_id, score, label in zip(query_ids, doc_ids, scores, labels):
            if query_id not in batch_results:
                batch_results[query_id] = {}

            # Store both score and label for each document
            # Use max score if document appears multiple times
            if doc_id not in batch_results[query_id] or score > batch_results[query_id][doc_id][0]:
                batch_results[query_id][doc_id] = [score, label]

        return batch_results

    def predict_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Get predictions for a single batch without organizing by query.

        Args:
            batch: Batch data

        Returns:
            Tensor of prediction scores
        """
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model(
                query_input_ids=batch['query_input_ids'],
                query_attention_mask=batch['query_attention_mask'],
                query_token_type_ids=batch['query_token_type_ids'],
                query_entity_emb=batch['query_entity_emb'],
                doc_input_ids=batch['doc_input_ids'],
                doc_attention_mask=batch['doc_attention_mask'],
                doc_token_type_ids=batch['doc_token_type_ids'],
                doc_entity_emb=batch['doc_entity_emb'],
                query_entity_mask=batch.get('query_entity_mask'),
                doc_entity_mask=batch.get('doc_entity_mask'),
                doc_scores=batch.get('doc_score')
            )

        return outputs['score']

    def predict_single(self,
                       query_text: str,
                       doc_text: str,
                       tokenizer,
                       query_entities: Optional[torch.Tensor] = None,
                       doc_entities: Optional[torch.Tensor] = None,
                       max_length: int = 512) -> float:
        """
        Predict score for a single query-document pair.

        Args:
            query_text: Query text
            doc_text: Document text
            tokenizer: Tokenizer for text processing
            query_entities: Query entity embeddings (optional)
            doc_entities: Document entity embeddings (optional)
            max_length: Maximum sequence length

        Returns:
            Prediction score
        """
        # Tokenize inputs
        query_inputs = tokenizer(
            query_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        doc_inputs = tokenizer(
            doc_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        # Prepare entity embeddings
        if query_entities is None:
            query_entities = torch.empty(1, 0, 300)
        if doc_entities is None:
            doc_entities = torch.empty(1, 0, 300)

        # Ensure proper shapes
        if len(query_entities.shape) == 2:
            query_entities = query_entities.unsqueeze(0)
        if len(doc_entities.shape) == 2:
            doc_entities = doc_entities.unsqueeze(0)

        # Move to device
        query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
        doc_inputs = {k: v.to(self.device) for k, v in doc_inputs.items()}
        query_entities = query_entities.to(self.device)
        doc_entities = doc_entities.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                query_input_ids=query_inputs['input_ids'],
                query_attention_mask=query_inputs['attention_mask'],
                query_token_type_ids=query_inputs.get('token_type_ids', torch.zeros_like(query_inputs['input_ids'])),
                query_entity_emb=query_entities,
                doc_input_ids=doc_inputs['input_ids'],
                doc_attention_mask=doc_inputs['attention_mask'],
                doc_token_type_ids=doc_inputs.get('token_type_ids', torch.zeros_like(doc_inputs['input_ids'])),
                doc_entity_emb=doc_entities
            )

        return outputs['score'].item()

    def get_embeddings(self, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Extract model embeddings for analysis.

        Args:
            data_loader: DataLoader with data

        Returns:
            Dictionary with embeddings
        """
        all_embeddings = []
        all_query_ids = []
        all_doc_ids = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting embeddings"):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Get model outputs including embeddings
                outputs = self.model(
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    query_token_type_ids=batch['query_token_type_ids'],
                    query_entity_emb=batch['query_entity_emb'],
                    doc_input_ids=batch['doc_input_ids'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    doc_token_type_ids=batch['doc_token_type_ids'],
                    doc_entity_emb=batch['doc_entity_emb'],
                    query_entity_mask=batch.get('query_entity_mask'),
                    doc_entity_mask=batch.get('doc_entity_mask'),
                    doc_scores=batch.get('doc_score')
                )

                # Store embeddings
                if 'combined_emb' in outputs:
                    all_embeddings.append(outputs['combined_emb'].cpu())
                all_query_ids.extend(batch['query_id'])
                all_doc_ids.extend(batch['doc_id'])

        return {
            'embeddings': torch.cat(all_embeddings, dim=0) if all_embeddings else None,
            'query_ids': all_query_ids,
            'doc_ids': all_doc_ids
        }

    def evaluate_with_metrics(self,
                              data_loader: DataLoader,
                              qrels_file: str,
                              metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model and compute TREC metrics.

        Args:
            data_loader: DataLoader with evaluation data
            qrels_file: Path to qrels file
            metrics: List of metrics to compute

        Returns:
            Dictionary with computed metrics
        """
        from .metrics import evaluate_run
        from .ranking_utils import save_trec_run
        import tempfile
        import os

        if metrics is None:
            metrics = ['map', 'ndcg', 'P_10', 'recip_rank']

        # Get predictions
        results = self.evaluate(data_loader)

        # Save to temporary run file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.run', delete=False) as f:
            temp_run_file = f.name
            save_trec_run(temp_run_file, results)

        try:
            # Compute metrics
            metric_results = {}
            for metric in metrics:
                metric_results[metric] = evaluate_run(qrels_file, temp_run_file, metric)

            return metric_results

        finally:
            # Clean up temporary file
            os.unlink(temp_run_file)

    def compare_models(self,
                       other_evaluator: 'QDERModelEvaluator',
                       data_loader: DataLoader) -> Dict[str, Any]:
        """
        Compare this model with another model.

        Args:
            other_evaluator: Another model evaluator
            data_loader: DataLoader with evaluation data

        Returns:
            Dictionary with comparison results
        """
        # Get predictions from both models
        results_1 = self.evaluate(data_loader)
        results_2 = other_evaluator.evaluate(data_loader)

        # Compare scores
        score_differences = []
        agreement_count = 0
        total_pairs = 0

        for query_id in results_1:
            if query_id in results_2:
                docs_1 = results_1[query_id]
                docs_2 = results_2[query_id]

                for doc_id in docs_1:
                    if doc_id in docs_2:
                        score_1 = docs_1[doc_id][0]
                        score_2 = docs_2[doc_id][0]

                        score_differences.append(score_1 - score_2)

                        # Check if models agree on relevance (using 0.5 threshold)
                        pred_1 = score_1 > 0.5
                        pred_2 = score_2 > 0.5
                        if pred_1 == pred_2:
                            agreement_count += 1

                        total_pairs += 1

        return {
            'num_compared_pairs': total_pairs,
            'mean_score_difference': sum(score_differences) / len(score_differences) if score_differences else 0,
            'score_difference_std': torch.tensor(score_differences).std().item() if score_differences else 0,
            'agreement_rate': agreement_count / total_pairs if total_pairs > 0 else 0,
            'score_differences': score_differences
        }

    def set_model(self, model: nn.Module) -> None:
        """
        Set a new model for evaluation.

        Args:
            model: New model to evaluate
        """
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        info = {
            'model_class': self.model.__class__.__name__,
            'device': self.device,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

        # Add model-specific info if available
        if hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())

        return info