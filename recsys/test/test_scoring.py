import torch
import pytest
from recsys.models.scoring import NCF, NES
from recsys.test.fixtures import (  # NOQA
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
    dummy_seq2seq_dataset,
    dummy_user_features,
)

@pytest.mark.parametrize(
    "model",
    [
        NCF,
        NES,
    ],
)
def test_train_and_score(model, dummy_interaction_dataset):
    model = model(dummy_interaction_dataset.data_schema)
    model.fit(dataset=dummy_interaction_dataset)

    pair = torch.tensor([[1, 2]])
    user = torch.tensor([[0, 1, 0, 1]])
    item = torch.tensor([[0, 0]])

    # Single score prediction.
    model.score(pair, user, item)

    # Batch score prediction.
    users = [1,2]
    items = [1,2,3]

    users_features = dummy_interaction_dataset.get_user_features(users)
    items_features = dummy_interaction_dataset.get_item_features(items)
    model.batch_score(users, items, users_features, items_features)