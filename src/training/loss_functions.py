"""
Loss functions for QDER model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss.

    Standard loss function for binary classification tasks.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        """
        Initialize BCE with logits loss.

        Args:
            pos_weight: Weight for positive examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE with logits loss.

        Args:
            predictions: Model predictions [batch_size]
            targets: Target labels [batch_size]

        Returns:
            Loss value
        """
        return self.loss_fn(predictions, targets)


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for learning to rank.

    Encourages relevant documents to have higher scores than irrelevant ones.
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize ranking loss.

        Args:
            margin: Margin for ranking loss
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Args:
            predictions: Model predictions [batch_size]
            targets: Target labels [batch_size] (0 or 1)

        Returns:
            Loss value
        """
        # Find positive and negative examples
        pos_mask = targets == 1
        neg_mask = targets == 0

        if not pos_mask.any() or not neg_mask.any():
            # If we don't have both positive and negative examples, use BCE
            return F.binary_cross_entropy_with_logits(predictions, targets.float())

        pos_scores = predictions[pos_mask]
        neg_scores = predictions[neg_mask]

        # Create all pairwise combinations
        pos_scores_expanded = pos_scores.unsqueeze(1)  # [num_pos, 1]
        neg_scores_expanded = neg_scores.unsqueeze(0)  # [1, num_neg]

        # Compute ranking loss: max(0, margin - (pos_score - neg_score))
        loss_matrix = torch.clamp(self.margin - (pos_scores_expanded - neg_scores_expanded), min=0)

        if self.reduction == 'mean':
            return loss_matrix.mean()
        elif self.reduction == 'sum':
            return loss_matrix.sum()
        else:
            return loss_matrix


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focuses learning on hard examples by down-weighting easy examples.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Model predictions [batch_size]
            targets: Target labels [batch_size]

        Returns:
            Loss value
        """
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')

        # Compute probabilities
        probs = torch.sigmoid(predictions)

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning similar and dissimilar pairs.
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize contrastive loss.

        Args:
            margin: Margin for dissimilar pairs
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings1: First set of embeddings [batch_size, emb_dim]
            embeddings2: Second set of embeddings [batch_size, emb_dim]
            targets: Target labels [batch_size] (1 for similar, 0 for dissimilar)

        Returns:
            Loss value
        """
        # Compute euclidean distance
        distances = F.pairwise_distance(embeddings1, embeddings2)

        # Compute contrastive loss
        pos_loss = targets.float() * distances.pow(2)
        neg_loss = (1 - targets.float()) * torch.clamp(self.margin - distances, min=0).pow(2)

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for learning embeddings with anchor, positive, and negative samples.
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, emb_dim]
            positive: Positive embeddings [batch_size, emb_dim]
            negative: Negative embeddings [batch_size, emb_dim]

        Returns:
            Loss value
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)

        # Compute triplet loss
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions with different weights.
    """

    def __init__(self, loss_functions: Dict[str, nn.Module], weights: Dict[str, float]):
        """
        Initialize combined loss.

        Args:
            loss_functions: Dictionary of loss functions
            weights: Weights for each loss function
        """
        super(CombinedLoss, self).__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.weights = weights

        # Validate weights
        for name in loss_functions.keys():
            if name not in weights:
                raise ValueError(f"Missing weight for loss function: {name}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            predictions: Model predictions
            targets: Target labels
            **kwargs: Additional arguments for specific loss functions

        Returns:
            Combined loss value
        """
        total_loss = 0.0

        for name, loss_fn in self.loss_functions.items():
            weight = self.weights[name]

            # Compute individual loss
            if name in kwargs:
                # Use specific arguments for this loss function
                loss_value = loss_fn(**kwargs[name])
            else:
                # Use default arguments
                loss_value = loss_fn(predictions, targets)

            total_loss += weight * loss_value

        return total_loss


# Loss function registry
LOSS_REGISTRY = {
    'bce': BCEWithLogitsLoss,
    'bce_logits': BCEWithLogitsLoss,
    'ranking': RankingLoss,
    'focal': FocalLoss,
    'contrastive': ContrastiveLoss,
    'triplet': TripletLoss,
    'combined': CombinedLoss
}


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance

    Raises:
        ValueError: If loss_name is not supported
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unsupported loss function: {loss_name}. "
                         f"Supported functions: {list(LOSS_REGISTRY.keys())}")

    loss_class = LOSS_REGISTRY[loss_name]
    return loss_class(**kwargs)


def create_weighted_bce_loss(positive_weight: float) -> BCEWithLogitsLoss:
    """
    Create weighted BCE loss for handling class imbalance.

    Args:
        positive_weight: Weight for positive class

    Returns:
        Weighted BCE loss
    """
    pos_weight = torch.tensor([positive_weight])
    return BCEWithLogitsLoss(pos_weight=pos_weight)


def create_combined_loss(primary_loss: str = 'bce',
                         auxiliary_loss: str = 'ranking',
                         primary_weight: float = 0.8,
                         auxiliary_weight: float = 0.2,
                         **kwargs) -> CombinedLoss:
    """
    Create a combined loss with primary and auxiliary components.

    Args:
        primary_loss: Name of primary loss function
        auxiliary_loss: Name of auxiliary loss function
        primary_weight: Weight for primary loss
        auxiliary_weight: Weight for auxiliary loss
        **kwargs: Additional arguments for loss functions

    Returns:
        Combined loss function
    """
    loss_functions = {
        'primary': get_loss_function(primary_loss, **kwargs.get('primary_kwargs', {})),
        'auxiliary': get_loss_function(auxiliary_loss, **kwargs.get('auxiliary_kwargs', {}))
    }

    weights = {
        'primary': primary_weight,
        'auxiliary': auxiliary_weight
    }

    return CombinedLoss(loss_functions, weights)