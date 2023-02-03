from torch.utils.data import DataLoader

from recsys.datasets import InteractionsDataset

from .fixtures import (  # NOQA
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
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
        user_id="user_id",
        item_id="item_id",
        sample_negatives=1,
    )
    dummy_interaction_dataset[0]
    dataloader = DataLoader(dummy_interaction_dataset)
    next(iter(dataloader))
