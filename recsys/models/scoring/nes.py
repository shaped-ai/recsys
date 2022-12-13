from typing import List

import torch
from torch import nn

from recsys.layers.utils import compute_similarity
from recsys.models.base import BaseModel
from recsys.models.trainers import PytorchLightningLiteTrainer
from recsys.models.utils import align_item_user_dimensions, schema_to_featureModuleList


class NES(nn.Module, BaseModel):
    """
    This class implements the Neural Embedding Similarity (NES) model.
    """

    def __init__(
        self,
        data_schema,
        lr_rate: float = 0.01,
        id_embedding_size: int = 64,
        feature_embedding_size: int = 8,
        mlp_layers: List[int] = None,
        similarity_function: str = "dot",
        accelerator: str = "cpu",
    ):
        super().__init__()

        user_id_dimension, item_id_dimension = align_item_user_dimensions(
            data_schema["user_features"],
            data_schema["item_features"],
            feature_embedding_size,
            id_embedding_size,
        )
        # User features encoding
        feature_embedding_sizes = {
            data_schema["user_id"]: user_id_dimension,
        }
        self.user_features, self.user_feature_dimension = schema_to_featureModuleList(
            data_schema["user_features"],
            feature_embedding_size,
            feature_embedding_sizes,
        )

        # Item features encoding
        feature_embedding_sizes = {
            data_schema["item_id"]: item_id_dimension,
        }
        self.item_features, self.item_feature_dimension = schema_to_featureModuleList(
            data_schema["item_features"],
            feature_embedding_size,
            feature_embedding_sizes,
        )

        n_users = data_schema["user_features"][
            data_schema["user_id"]
        ].unique_value_count
        n_items = data_schema["item_features"][
            data_schema["item_id"]
        ].unique_value_count

        # Find user_feature idx - TODO: IMPROVE
        self._user_id_feature_idx = [
            i
            for i, f in enumerate(self.user_features)
            if f.name == data_schema["user_id"]
        ][0]
        self._item_id_feature_idx = [
            i
            for i, f in enumerate(self.item_features)
            if f.name == data_schema["item_id"]
        ][0]

        self.user_bias = nn.Embedding(n_users + 1, 1)
        self.item_bias = nn.Embedding(n_items + 1, 1)

        # User mlp
        user_mlp_layers = (
            [user_id_dimension]
            if mlp_layers is None
            else [user_id_dimension] + mlp_layers
        )

        self.user_mlp = torch.nn.Sequential(
            *[
                nn.Linear(user_mlp_layers[i], user_mlp_layers[i + 1])
                for i in range(0, len(user_mlp_layers) - 1)
            ]
        )

        item_mlp_layers = (
            [item_id_dimension]
            if mlp_layers is None
            else [item_id_dimension] + mlp_layers
        )

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

    def forward(self, users_features, items_features):
        user_vector = self.encode_user(users_features)
        item_vector = self.encode_item(items_features)

        yhat = self.score(user_vector, item_vector)

        # Need to figure out how to compute biases also for pure raw vectors nicely.
        print(self.user_bias)
        print(users_features[:, self._user_id_feature_idx])
        user_bias = torch.squeeze(
            self.user_bias(users_features[:, self._user_id_feature_idx])
        )

        print(self.item_bias)
        print(items_features[:, self._item_id_feature_idx])
        item_bias = torch.squeeze(
            self.item_bias(items_features[:, self._item_id_feature_idx])
        )

        yhat = yhat + user_bias + item_bias
        return yhat

    def score(self, user_vector, item_vector):
        user_vector = torch.as_tensor(user_vector)
        item_vector = torch.as_tensor(item_vector)

        user_vector = self.user_mlp(user_vector)
        item_vector = self.item_mlp(item_vector)

        yhat = compute_similarity(
            x=user_vector, y=item_vector, mode=self.similarity_function
        )
        return yhat

    def batch_score(self, users, items, batch_size=256):
        r = []
        for i, user in enumerate(users):
            single_user_scores = []
            # Create pairs of the user with all items
            single_user_features = user.repeat(len(items), 1)
            for ndx in range(0, len(items), batch_size):
                single_user_scores.append(
                    self.score(
                        single_user_features[ndx : ndx + batch_size],
                        items[ndx : ndx + batch_size],
                    )
                )

            r.append(torch.cat(single_user_scores))

        return torch.stack(r)

    def generate_item_matrix(self, items):
        return self.item_embedding(items)

    def training_step(self, batch):
        interactions, users, items = batch

        yhat = self(users, items).float()
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
