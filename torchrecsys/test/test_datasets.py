from torch.utils.data import DataLoader

from torchrecsys.datasets import InteractionsDataset, Seq2SeqDataset
from torchrecsys.test.fixtures import (  # NOQA
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
    dummy_seq2seq_dataset,
    dummy_user_features,
)


def test_interactions_dataset(
    dummy_interactions, dummy_user_features, dummy_item_features
):
    dummy_interaction_dataset = InteractionsDataset(
        dummy_interactions[["user_id", "item_id", "rating"]],
        dummy_user_features,
        dummy_item_features,
        interaction_id="rating",
        sample_negatives=1,
    )
    dummy_interaction_dataset[0]
    dataloader = DataLoader(dummy_interaction_dataset)
    next(iter(dataloader))


def test_sequence_dataset(dummy_interactions, dummy_user_features, dummy_item_features):
    sorted_sequences = (
        dummy_interactions.sort_values(by=["timestamp"])
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_frame()
    )
    dummy_seq2seq_dataset = Seq2SeqDataset(
        sorted_sequences,
        dummy_item_features,
        sequence_id="item_id",
        item_id="item_id",
        max_item_id=int(dummy_item_features.item_id.max()),
    )
    dummy_seq2seq_dataset[0]
    dataloader = DataLoader(dummy_seq2seq_dataset)
    next(iter(dataloader))
