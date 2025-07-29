"""
DataLoader wrapper for QDER datasets.
"""

from torch.utils.data import DataLoader
from .dataset import QDERDataset


class QDERDataLoader(DataLoader):
    """
    DataLoader wrapper for QDERDataset.

    Provides a convenient interface for loading QDER data with proper
    collation and batching functionality.
    """

    def __init__(self,
                 dataset: QDERDataset,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 sampler=None,
                 drop_last: bool = False,
                 pin_memory: bool = False) -> None:
        """
        Initialize the DataLoader.

        Args:
            dataset: QDERDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data (ignored if sampler is provided)
            num_workers: Number of worker processes for data loading
            sampler: Optional sampler for custom sampling strategy
            drop_last: Whether to drop the last incomplete batch
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            collate_fn=dataset.collate,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=pin_memory
        )

        self.dataset = dataset

    def get_dataset_info(self) -> dict:
        """
        Get information about the underlying dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_examples': len(self.dataset),
            'max_length': self.dataset._max_len,
            'is_training': self.dataset._train
        }