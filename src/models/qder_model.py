"""
Main QDER model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base_model import BaseQDERModel
from .text_embedding import TextEmbedding


class QDERModel(BaseQDERModel):
    """
    Query-Document Entity Ranking (QDER) model.

    This is the main model from your original model.py file, cleaned up and
    integrated into the modular structure.
    """

    def __init__(self,
                 pretrained: str,
                 use_scores: bool = True,
                 use_entities: bool = True,
                 score_method: str = 'linear') -> None:
        """
        Initialize QDER model.

        Args:
            pretrained: Name of pretrained text encoder
            use_scores: Whether to use document retrieval scores
            use_entities: Whether to use entity embeddings
            score_method: Scoring method ('linear' or 'bilinear')
        """
        super(QDERModel, self).__init__(pretrained, use_scores, use_entities, score_method)

        # Text encoders
        self.query_encoder = TextEmbedding(pretrained=pretrained)
        self.doc_encoder = TextEmbedding(pretrained=pretrained)

        # Get embedding dimensions
        text_dim = self.query_encoder.get_embedding_dim()  # 768 for BERT
        entity_dim = 300  # Entity embedding dimension

        # Calculate total feature dimension
        total_dim = 3 * text_dim  # sub, add, mul interactions for text
        if use_entities:
            total_dim += 3 * entity_dim  # sub, add, mul interactions for entities

        # Scoring layer
        if score_method == 'linear':
            self.score = nn.Linear(in_features=total_dim, out_features=1)
        elif score_method == 'bilinear':
            self.score = nn.Bilinear(
                in1_features=total_dim,
                in2_features=total_dim,
                out_features=1
            )
        else:
            raise ValueError(f"Unknown score_method: {score_method}")

    def forward(self,
                query_input_ids: torch.Tensor,
                query_attention_mask: torch.Tensor,
                query_token_type_ids: torch.Tensor,
                query_entity_emb: torch.Tensor,
                doc_input_ids: torch.Tensor,
                doc_attention_mask: torch.Tensor,
                doc_token_type_ids: torch.Tensor,
                doc_entity_emb: torch.Tensor,
                query_entity_mask: Optional[torch.Tensor] = None,
                doc_entity_mask: Optional[torch.Tensor] = None,
                doc_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through QDER model.

        Args:
            query_input_ids: Query token IDs [batch_size, seq_len]
            query_attention_mask: Query attention mask [batch_size, seq_len]
            query_token_type_ids: Query token type IDs [batch_size, seq_len]
            query_entity_emb: Query entity embeddings [batch_size, num_q_entities, 300]
            doc_input_ids: Document token IDs [batch_size, seq_len]
            doc_attention_mask: Document attention mask [batch_size, seq_len]
            doc_token_type_ids: Document token type IDs [batch_size, seq_len]
            doc_entity_emb: Document entity embeddings [batch_size, num_d_entities, 300]
            query_entity_mask: Query entity attention mask [batch_size, num_q_entities]
            doc_entity_mask: Document entity attention mask [batch_size, num_d_entities]
            doc_scores: Document retrieval scores [batch_size]

        Returns:
            Dictionary containing 'score' and 'combined_emb'
        """
        # Get text embeddings
        query_text_emb = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids
        )
        doc_text_emb = self.doc_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            token_type_ids=doc_token_type_ids
        )

        # Text-based attention
        text_attention_scores = torch.matmul(query_text_emb, doc_text_emb.transpose(-2, -1))
        text_attention_probs = F.softmax(text_attention_scores, dim=-1)
        weighted_doc_text_emb = torch.matmul(text_attention_probs, doc_text_emb)

        # Compute text interactions
        text_emb_sub = torch.sub(input=query_text_emb, other=weighted_doc_text_emb, alpha=1)
        text_emb_add = torch.add(input=query_text_emb, other=weighted_doc_text_emb, alpha=1)
        text_emb_mul = query_text_emb * weighted_doc_text_emb

        # Mean pooling for text embeddings
        text_emb_sub_pool = torch.mean(text_emb_sub, dim=1)
        text_emb_add_pool = torch.mean(text_emb_add, dim=1)
        text_emb_mul_pool = torch.mean(text_emb_mul, dim=1)

        # Entity-based interactions (if enabled)
        if self.use_entities and query_entity_emb.size(1) > 0 and doc_entity_emb.size(1) > 0:
            # Entity-based attention
            entity_attention_scores = torch.matmul(query_entity_emb, doc_entity_emb.transpose(-2, -1))
            entity_attention_probs = F.softmax(entity_attention_scores, dim=-1)
            weighted_doc_entity_emb = torch.matmul(entity_attention_probs, doc_entity_emb)

            # Compute entity interactions
            entity_emb_sub = torch.sub(input=query_entity_emb, other=weighted_doc_entity_emb, alpha=1)
            entity_emb_add = torch.add(input=query_entity_emb, other=weighted_doc_entity_emb, alpha=1)
            entity_emb_mul = query_entity_emb * weighted_doc_entity_emb

            # Mean pooling for entity embeddings (with masking if available)
            if query_entity_mask is not None:
                # Apply mask before pooling
                query_mask_expanded = query_entity_mask.unsqueeze(-1).float()
                entity_emb_sub = entity_emb_sub * query_mask_expanded
                entity_emb_add = entity_emb_add * query_mask_expanded
                entity_emb_mul = entity_emb_mul * query_mask_expanded

                # Compute masked mean
                mask_sum = query_entity_mask.sum(dim=1, keepdim=True).float()
                mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
                entity_emb_sub_pool = torch.sum(entity_emb_sub, dim=1) / mask_sum
                entity_emb_add_pool = torch.sum(entity_emb_add, dim=1) / mask_sum
                entity_emb_mul_pool = torch.sum(entity_emb_mul, dim=1) / mask_sum
            else:
                # Simple mean pooling
                entity_emb_sub_pool = torch.mean(entity_emb_sub, dim=1)
                entity_emb_add_pool = torch.mean(entity_emb_add, dim=1)
                entity_emb_mul_pool = torch.mean(entity_emb_mul, dim=1)
        else:
            # Create zero entity embeddings if not using entities
            batch_size = query_text_emb.size(0)
            entity_dim = 300
            device = query_text_emb.device
            entity_emb_sub_pool = torch.zeros(batch_size, entity_dim, device=device)
            entity_emb_add_pool = torch.zeros(batch_size, entity_dim, device=device)
            entity_emb_mul_pool = torch.zeros(batch_size, entity_dim, device=device)

        # Apply document scores if enabled
        if self.use_scores and doc_scores is not None:
            doc_score_expanded = doc_scores.unsqueeze(-1)
            text_emb_sub_pool = doc_score_expanded * text_emb_sub_pool
            text_emb_add_pool = doc_score_expanded * text_emb_add_pool
            text_emb_mul_pool = doc_score_expanded * text_emb_mul_pool

            if self.use_entities:
                entity_emb_sub_pool = doc_score_expanded * entity_emb_sub_pool
                entity_emb_add_pool = doc_score_expanded * entity_emb_add_pool
                entity_emb_mul_pool = doc_score_expanded * entity_emb_mul_pool

        # Concatenate all interaction features
        if self.use_entities:
            combined_emb = torch.cat([
                text_emb_sub_pool, text_emb_add_pool, text_emb_mul_pool,
                entity_emb_sub_pool, entity_emb_add_pool, entity_emb_mul_pool
            ], dim=-1)
        else:
            combined_emb = torch.cat([
                text_emb_sub_pool, text_emb_add_pool, text_emb_mul_pool
            ], dim=-1)

        # Pass through scoring layer
        if self.score_method == 'linear':
            score = self.score(combined_emb)
        else:  # bilinear
            score = self.score(combined_emb, combined_emb)

        return {
            'score': score.squeeze(dim=-1),
            'combined_emb': combined_emb
        }

    def get_text_embeddings(self,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            token_type_ids: torch.Tensor,
                            encoder_type: str = 'query') -> torch.Tensor:
        """
        Get text embeddings from the specified encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            encoder_type: 'query' or 'doc' encoder

        Returns:
            Text embeddings [batch_size, seq_len, hidden_size]
        """
        if encoder_type == 'query':
            return self.query_encoder(input_ids, attention_mask, token_type_ids)
        elif encoder_type == 'doc':
            return self.doc_encoder(input_ids, attention_mask, token_type_ids)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def get_interaction_features(self,
                                 query_text_emb: torch.Tensor,
                                 doc_text_emb: torch.Tensor,
                                 query_entity_emb: Optional[torch.Tensor] = None,
                                 doc_entity_emb: Optional[torch.Tensor] = None,
                                 query_entity_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract interaction features between query and document representations.

        Args:
            query_text_emb: Query text embeddings
            doc_text_emb: Document text embeddings
            query_entity_emb: Query entity embeddings (optional)
            doc_entity_emb: Document entity embeddings (optional)
            query_entity_mask: Query entity mask (optional)

        Returns:
            Combined interaction features
        """
        # Text-based attention
        text_attention_scores = torch.matmul(query_text_emb, doc_text_emb.transpose(-2, -1))
        text_attention_probs = F.softmax(text_attention_scores, dim=-1)
        weighted_doc_text_emb = torch.matmul(text_attention_probs, doc_text_emb)

        # Compute text interactions
        text_emb_sub = torch.sub(input=query_text_emb, other=weighted_doc_text_emb, alpha=1)
        text_emb_add = torch.add(input=query_text_emb, other=weighted_doc_text_emb, alpha=1)
        text_emb_mul = query_text_emb * weighted_doc_text_emb

        # Mean pooling for text embeddings
        text_emb_sub_pool = torch.mean(text_emb_sub, dim=1)
        text_emb_add_pool = torch.mean(text_emb_add, dim=1)
        text_emb_mul_pool = torch.mean(text_emb_mul, dim=1)

        # Entity interactions if enabled and available
        if (self.use_entities and query_entity_emb is not None and doc_entity_emb is not None and
                query_entity_emb.size(1) > 0 and doc_entity_emb.size(1) > 0):

            # Entity-based attention
            entity_attention_scores = torch.matmul(query_entity_emb, doc_entity_emb.transpose(-2, -1))
            entity_attention_probs = F.softmax(entity_attention_scores, dim=-1)
            weighted_doc_entity_emb = torch.matmul(entity_attention_probs, doc_entity_emb)

            # Compute entity interactions
            entity_emb_sub = torch.sub(input=query_entity_emb, other=weighted_doc_entity_emb, alpha=1)
            entity_emb_add = torch.add(input=query_entity_emb, other=weighted_doc_entity_emb, alpha=1)
            entity_emb_mul = query_entity_emb * weighted_doc_entity_emb

            # Mean pooling for entity embeddings (with masking if available)
            if query_entity_mask is not None:
                query_mask_expanded = query_entity_mask.unsqueeze(-1).float()
                entity_emb_sub = entity_emb_sub * query_mask_expanded
                entity_emb_add = entity_emb_add * query_mask_expanded
                entity_emb_mul = entity_emb_mul * query_mask_expanded

                mask_sum = query_entity_mask.sum(dim=1, keepdim=True).float()
                mask_sum = torch.clamp(mask_sum, min=1.0)
                entity_emb_sub_pool = torch.sum(entity_emb_sub, dim=1) / mask_sum
                entity_emb_add_pool = torch.sum(entity_emb_add, dim=1) / mask_sum
                entity_emb_mul_pool = torch.sum(entity_emb_mul, dim=1) / mask_sum
            else:
                entity_emb_sub_pool = torch.mean(entity_emb_sub, dim=1)
                entity_emb_add_pool = torch.mean(entity_emb_add, dim=1)
                entity_emb_mul_pool = torch.mean(entity_emb_mul, dim=1)

            # Concatenate text and entity features
            combined_features = torch.cat([
                text_emb_sub_pool, text_emb_add_pool, text_emb_mul_pool,
                entity_emb_sub_pool, entity_emb_add_pool, entity_emb_mul_pool
            ], dim=-1)
        else:
            # Only text features
            combined_features = torch.cat([
                text_emb_sub_pool, text_emb_add_pool, text_emb_mul_pool
            ], dim=-1)

        return combined_features

    def compute_similarity_score(self,
                                 query_text: str,
                                 doc_text: str,
                                 tokenizer,
                                 query_entities: Optional[torch.Tensor] = None,
                                 doc_entities: Optional[torch.Tensor] = None,
                                 max_length: int = 512) -> float:
        """
        Compute similarity score for a single query-document pair.

        Args:
            query_text: Query text string
            doc_text: Document text string
            tokenizer: Tokenizer for text processing
            query_entities: Query entity embeddings (optional)
            doc_entities: Document entity embeddings (optional)
            max_length: Maximum sequence length

        Returns:
            Similarity score as float
        """
        self.eval()

        with torch.no_grad():
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

            # Forward pass
            output = self.forward(
                query_input_ids=query_inputs['input_ids'],
                query_attention_mask=query_inputs['attention_mask'],
                query_token_type_ids=query_inputs.get('token_type_ids', torch.zeros_like(query_inputs['input_ids'])),
                query_entity_emb=query_entities,
                doc_input_ids=doc_inputs['input_ids'],
                doc_attention_mask=doc_inputs['attention_mask'],
                doc_token_type_ids=doc_inputs.get('token_type_ids', torch.zeros_like(doc_inputs['input_ids'])),
                doc_entity_emb=doc_entities
            )

            return output['score'].item()