from torch.utils.data import DataLoader

from torchrecsys.test.fixtures import (  # NOQA
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
    dummy_seq2seq_dataset,
    dummy_user_features,
)


def test_interactions_dataset(dummy_interaction_dataset):
    dummy_interaction_dataset[0]
    dataloader = DataLoader(dummy_interaction_dataset)
    next(iter(dataloader))


def test_sequence_dataset(dummy_seq2seq_dataset):
    dummy_seq2seq_dataset[0]
    dataloader = DataLoader(dummy_seq2seq_dataset)
    next(iter(dataloader))
