from typing import List

import torch
from torch import nn

from recsys.layers.utils import compute_similarity
from recsys.models.base import BaseModel
from recsys.models.trainers import PytorchLightningLiteTrainer
from recsys.models.utils import schema_to_featureModuleList


class NES(nn.Module, BaseModel):
    """
    This class implements the Neural Embedding Similarity (NES) model.
    """
    def __init__(
        self,
        data_schema,
        lr_rate: float = 0.01,
        embedding_size: int = 64,
        feature_embedding_size: int = 8,
        mlp_layers: List[int] = None,  # NOQA B006
        similarity_function: str = "dot",
        accelerator: str = "cpu",
    ):
        super().__init__()

        interactions_schema = data_schema["interactions"]

        self.n_users = interactions_schema[0]
        self.n_items = interactions_schema[1]

        # User features encoding
        self.user_features, self.user_feature_dimension = schema_to_featureModuleList(
            data_schema["user_features"], feature_embedding_size
        )

        # Item features encoding
        self.item_features, self.item_feature_dimension = schema_to_featureModuleList(
            data_schema["item_features"], feature_embedding_size
        )

        self.user_embedding = nn.Embedding(self.n_users + 1, embedding_size)
        self.item_embedding = nn.Embedding(self.n_items + 1, embedding_size)

        self.user_bias = nn.Embedding(self.n_users + 1, 1)
        self.item_bias = nn.Embedding(self.n_items + 1, 1)


        total_user_dimension = embedding_size + self.user_feature_dimension
        total_item_dimension = embedding_size + self.item_feature_dimension

        dominant_embedding_size = max(total_user_dimension, total_item_dimension)
        mlp_layers = [dominant_embedding_size] if mlp_layers is None else mlp_layers

        # User mlp
        user_mlp_layers = [self.user_feature_dimension + embedding_size] + mlp_layers

        self.user_mlp = torch.nn.Sequential(
            *[
                nn.Linear(user_mlp_layers[i], user_mlp_layers[i + 1])
                for i in range(0, len(user_mlp_layers) - 1)
            ]
        )

        item_mlp_layers = [self.item_feature_dimension + embedding_size] + mlp_layers

        self.item_mlp = torch.nn.Sequential(
            *[
                nn.Linear(item_mlp_layers[i], item_mlp_layers[i + 1])
                for i in range(0, len(item_mlp_layers) - 1)
            ]
        )

        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if data_schema["objetive"] == "binary"
            else torch.nn.MSELoss()
        )

        self.lr_rate = lr_rate
        self.similarity_function = similarity_function

        # Trainer
        self._trainer = PytorchLightningLiteTrainer(accelerator=accelerator)

    def forward(self, interactions, users_features, items_features):
        user = interactions[:, 0].long()
        item = interactions[:, 1].long()

        user_factor = self.user_embedding(user)
        item_factor = self.item_embedding(item)

        user_features = self.encode_user(users_features)
        item_features = self.encode_item(items_features)

        user_factor = torch.cat([user_factor, user_features], dim=1)
        item_factor = torch.cat([item_factor, item_features], dim=1)

        user_factor = self.user_mlp(user_factor)
        item_factor = self.item_mlp(item_factor)

        user_bias = torch.squeeze(self.user_bias(user))
        item_bias = torch.squeeze(self.item_bias(item))
        yhat = (
            compute_similarity(
                x=user_factor, y=item_factor, mode=self.similarity_function
            )
            + user_bias
            + item_bias
        )

        return yhat

    def encode_user(self, user):
        r = []
        for _idx, feature in enumerate(self.user_features):
            feature_embedding = feature(user[:, feature.idx])
            r.append(feature_embedding)
        r = torch.cat(r, dim=1)
        return r

    def encode_item(self, item):
        r = []
        for _idx, feature in enumerate(self.item_features):
            feature_embedding = feature(item[:, feature.idx])
            r.append(feature_embedding)
        r = torch.cat(r, dim=1)
        return r

    def score(self, pair, user_features, item_features):
        pair = torch.as_tensor(pair)
        user_features = torch.as_tensor(user_features)
        item_features = torch.as_tensor(item_features)
        return self(pair, user_features, item_features)

    def batch_score(self, users, items, user_features, item_features, batch_size=256):
        r = []
        for i, user in enumerate(users):
            single_user_scores = []
            # Create pairs of the user with all items
            single_user_features = user_features[i].repeat(len(items), 1)
            pairs = torch.tensor([[user, item] for item in items])
            for ndx in range(0, len(pairs), batch_size):
                single_user_scores.append(
                    self.score(
                        pairs[ndx : ndx + batch_size],
                        single_user_features[ndx : ndx + batch_size],
                        item_features[ndx : ndx + batch_size],
                    )
                )

            r.append(torch.cat(single_user_scores))

        return torch.stack(r)

    def generate_item_matrix(self, items):
        return self.item_embedding(items)

    def training_step(self, batch):
        interactions, users, items = batch

        yhat = self(interactions.long(), users, items).float()
        ytrue = batch[0][:, 2].float()

        loss = self.criterion(yhat, ytrue)

        return loss

    def validation_step(self, batch):
        yhat = self(*batch).float()
        ytrue = batch[0][:, 2].float()
        loss = self.criterion(yhat, ytrue)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr_rate)

        return optimizer

    def generate_item_representations(self, interactions_dataset):
        # TODO MAKE this run under pl and optimize loop
        with torch.no_grad():
            torch.set_grad_enabled(False)
            self.eval()
            item_feature_dict = interactions_dataset.item_features
            items = []
            items_features = []
            for item in item_feature_dict.keys():
                items.append(item)
                items_features.append(item_feature_dict[item])
            items = torch.tensor(items)
            items_features = torch.tensor(items_features)

            item_embeddings = self.item_embedding(items)
            item_features = self.encode_item(items_features)

            item_embeddings = torch.cat([item_embeddings, item_features], dim=1)
            item_embeddings = self.item_mlp(item_embeddings)

        return torch.tensor(list(item_feature_dict.keys())), item_embeddings

    def generate_user_representations(self, interactions_dataset):
        # TODO MAKE this run under pl and optimize loop.
        with torch.no_grad():
            user_feature_dict = interactions_dataset.user_features
            users = []
            users_features = []
            for user in user_feature_dict.keys():
                users.append(user)
                users_features.append(user_feature_dict[user])
            users = torch.tensor(users)
            users_features = torch.tensor(users_features)

            user_embeddings = self.user_embedding(users)
            user_features = self.encode_user(users_features)

            user_embeddings = torch.cat([user_embeddings, user_features], dim=1)
            user_embeddings = self.user_mlp(user_embeddings)

        return torch.tensor(list(user_feature_dict.keys())), user_embeddings
