import torch
import pytest
from recsys.models.scoring import NCF, NES
from .fixtures import (  # NOQA
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
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

    test_users = dummy_interaction_dataset.get_user_features([1,2])
    test_items = dummy_interaction_dataset.get_item_features([1,2])

    test_users = model.encode_user(test_users)
    test_items = model.encode_item(test_items)

    # Single score prediction.
    results = model.score(test_users, test_items)
    assert list(results.shape) == [2]

    # Batch score prediction.
    results = model.batch_score(test_users, test_items)
    assert list(results.shape) == [2, 2]