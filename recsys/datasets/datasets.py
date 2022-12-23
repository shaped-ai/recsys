import numpy as np
import torch

from recsys.datasets.utils import dataframe_schema


class InteractionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        interactions,
        user_features,
        item_features,
        user_id="user_id",
        item_id="item_id",
        interaction_id="interaction",
        sample_negatives=0,
    ):
        _check_inputs(interactions, user_features, item_features, item_id, user_id)

        self.interactions = interactions[[user_id, item_id, interaction_id]]
        self.user_id = user_id
        self.item_id = item_id
        self.interaction_id = interaction_id

        self.user_features = dict(
            zip(
                user_features[self.user_id],
                user_features.to_numpy(dtype=int),
            )
        )
        self.item_features = dict(
            zip(
                item_features[self.item_id],
                item_features.to_numpy(dtype=int),
            )
        )

        self.interactions_pd_schema = dataframe_schema(self.interactions)
        self.user_pd_schema = dataframe_schema(user_features)
        self.item_pd_schema = dataframe_schema(item_features)

        if self.interactions[self.interaction_id].isin([0, 1]).all():
            self.target_type = "binary"
        else:
            self.target_type = "continuous"

        self.interactions = self.interactions.values
        self.sample_negatives = sample_negatives
        self.unique_items = item_features[self.item_id].unique()

    def __len__(self):
        return len(self.interactions)

    def _sample_negative(self):
        # TODO make this not sample already interacted items
        return np.random.choice(self.unique_items)

    def get_user_features(self, user_ids):
        return torch.tensor([self.user_features[user_id] for user_id in user_ids])

    def get_item_features(self, item_ids):
        return torch.tensor([self.item_features[item_id] for item_id in item_ids])

    def __getitem__(self, idx):
        interaction = self.interactions[idx]

        user, item, target = interaction[0], interaction[1], interaction[2]
        if self.sample_negatives > 0:
            negative_dice = np.random.randint(1, self.sample_negatives + 1)
            if negative_dice > 1:
                item = self._sample_negative()
                target = 0

        user_features = self.user_features[user]
        item_features = self.item_features[item]

        return (
            torch.tensor([user, item, target]),
            torch.tensor(user_features),
            torch.tensor(item_features),
        )

    @property
    def data_schema(self):
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "user_features": self.user_pd_schema,
            "item_features": self.item_pd_schema,
            "objetive": self.target_type,
        }


def _check_inputs(interactions, users, items, item_id, user_id):
    """
    Perform healths checks on the input data for the dataset
    """
    # Id columns exists
    assert item_id in items.columns
    assert user_id in users.columns
    assert item_id in interactions.columns
    assert user_id in interactions.columns

    # Id columns are unique for item/user
    assert len(items[item_id].unique()) == len(items)
    assert len(users[user_id].unique()) == len(users)

    # All users and items in interactions are in the users and items dataframes
    assert interactions[item_id].isin(items[item_id]).all()
    assert interactions[user_id].isin(users[user_id]).all()
