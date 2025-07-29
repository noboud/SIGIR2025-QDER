"""
QDER model with ablation capabilities for interaction analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .qder_model import QDERModel


class QDERAblation(QDERModel):
    """
    QDER model with configurable interaction ablations.

    Allows enabling/disabling specific interaction types (add, subtract, multiply)
    for ablation studies and analysis.
    """

    def __init__(self,
                 pretrained: str,
                 use_scores: bool = True,
                 use_entities: bool = True,
                 score_method: str = 'linear',
                 enabled_interactions: List[str] = None) -> None:
        """
        Initialize QDER ablation model.

        Args:
            pretrained: Name of pretrained text encoder
            use_scores: Whether to use document retrieval scores
            use_entities: Whether to use entity embeddings
            score_method: Scoring method ('linear' or 'bilinear')
            enabled_interactions: List of enabled interactions ['add', 'subtract', 'multiply']
        """
        super(QDERAblation, self).__init__(pretrained, use_scores, use_entities, score_method)

        if enabled_interactions is None:
            enabled_interactions = ['add', 'subtract', 'multiply']
        self.enabled_interactions = enabled_interactions

        # Update config to include ablation info
        self.config['enabled_interactions'] = enabled_interactions

        # Validate interactions
        valid_interactions = {'add', 'subtract', 'multiply'}
        for interaction in enabled_interactions:
            if interaction not in valid_interactions:
                raise ValueError(f"Invalid interaction: {interaction}. Must be one of {valid_interactions}")

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
        Forward pass with interaction ablations.

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

        # Initialize text interaction embeddings (zeros for disabled interactions)
        batch_size, seq_len, hidden_dim = query_text_emb.shape
        device = query_text_emb.device

        text_emb_sub = torch.zeros_like(query_text_emb)
        text_emb_add = torch.zeros_like(query_text_emb)
        text_emb_mul = torch.zeros_like(query_text_emb)

        # Compute enabled text interactions
        if 'subtract' in self.enabled_interactions:
            text_emb_sub = torch.sub(input=query_text_emb, other=weighted_doc_text_emb, alpha=1)
        if 'add' in self.enabled_interactions:
            text_emb_add = torch.add(input=query_text_emb, other=weighted_doc_text_emb, alpha=1)
        if 'multiply' in self.enabled_interactions:
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

            # Initialize entity interaction embeddings
            entity_emb_sub = torch.zeros_like(query_entity_emb)
            entity_emb_add = torch.zeros_like(query_entity_emb)
            entity_emb_mul = torch.zeros_like(query_entity_emb)

            # Compute enabled entity interactions
            if 'subtract' in self.enabled_interactions:
                entity_emb_sub = torch.sub(input=query_entity_emb, other=weighted_doc_entity_emb, alpha=1)
            if 'add' in self.enabled_interactions:
                entity_emb_add = torch.add(input=query_entity_emb, other=weighted_doc_entity_emb, alpha=1)
            if 'multiply' in self.enabled_interactions:
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
        else:
            # Create zero entity embeddings if not using entities
            entity_dim = 300
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

        # Concatenate all interaction features (zeros for disabled interactions)
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
            'combined_emb': combined_emb,
            'text_interactions': {
                'subtract': text_emb_sub_pool,
                'add': text_emb_add_pool,
                'multiply': text_emb_mul_pool
            },
            'entity_interactions': {
                'subtract': entity_emb_sub_pool,
                'add': entity_emb_add_pool,
                'multiply': entity_emb_mul_pool
            } if self.use_entities else None
        }

    def set_enabled_interactions(self, interactions: List[str]) -> None:
        """
        Update the enabled interactions.

        Args:
            interactions: List of interaction types to enable
        """
        valid_interactions = {'add', 'subtract', 'multiply'}
        for interaction in interactions:
            if interaction not in valid_interactions:
                raise ValueError(f"Invalid interaction: {interaction}. Must be one of {valid_interactions}")

        self.enabled_interactions = interactions
        self.config['enabled_interactions'] = interactions

    def get_enabled_interactions(self) -> List[str]:
        """Get the currently enabled interactions."""
        return self.enabled_interactions.copy()

    def disable_interaction(self, interaction: str) -> None:
        """
        Disable a specific interaction type.

        Args:
            interaction: Interaction type to disable
        """
        if interaction in self.enabled_interactions:
            self.enabled_interactions.remove(interaction)
            self.config['enabled_interactions'] = self.enabled_interactions

    def enable_interaction(self, interaction: str) -> None:
        """
        Enable a specific interaction type.

        Args:
            interaction: Interaction type to enable
        """
        valid_interactions = {'add', 'subtract', 'multiply'}
        if interaction not in valid_interactions:
            raise ValueError(f"Invalid interaction: {interaction}. Must be one of {valid_interactions}")

        if interaction not in self.enabled_interactions:
            self.enabled_interactions.append(interaction)
            self.config['enabled_interactions'] = self.enabled_interactions

    def get_interaction_contributions(self,
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
        Analyze the contribution of each interaction type to the final score.

        Returns scores with each interaction type individually enabled.
        """
        original_interactions = self.enabled_interactions.copy()
        contributions = {}

        # Test each interaction individually
        for interaction in ['add', 'subtract', 'multiply']:
            self.set_enabled_interactions([interaction])

            with torch.no_grad():
                output = self.forward(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    query_token_type_ids=query_token_type_ids,
                    query_entity_emb=query_entity_emb,
                    doc_input_ids=doc_input_ids,
                    doc_attention_mask=doc_attention_mask,
                    doc_token_type_ids=doc_token_type_ids,
                    doc_entity_emb=doc_entity_emb,
                    query_entity_mask=query_entity_mask,
                    doc_entity_mask=doc_entity_mask,
                    doc_scores=doc_scores
                )
                contributions[interaction] = output['score']

        # Test with no interactions
        self.set_enabled_interactions([])
        with torch.no_grad():
            output = self.forward(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                query_token_type_ids=query_token_type_ids,
                query_entity_emb=query_entity_emb,
                doc_input_ids=doc_input_ids,
                doc_attention_mask=doc_attention_mask,
                doc_token_type_ids=doc_token_type_ids,
                doc_entity_emb=doc_entity_emb,
                query_entity_mask=query_entity_mask,
                doc_entity_mask=doc_entity_mask,
                doc_scores=doc_scores
            )
            contributions['none'] = output['score']

        # Restore original interactions
        self.set_enabled_interactions(original_interactions)

        return contributions