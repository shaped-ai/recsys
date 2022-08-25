from typing import List

import torch
from torch import nn

from torchrecsys.models.base import BaseModel
from torchrecsys.models.trainers import PytorchLightningLiteTrainer
from torchrecsys.models.utils import schema_to_featureModuleList


class DeepRetriever(nn.Module, BaseModel):
    def __init__(
        self,
        data_schema,
        lr_rate: float = 0.01,
        embedding_size: int = 64,
        feature_embedding_size: int = 8,
        mlp_layers: List[int] = [512, 256],  # NOQA B006
        similarity_function: str = "dot",
        accelerator: str = "cpu",
    ):
        super().__init__()

        # TODO DATA SCHEMA CHECKS
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
        # User mlp
        mlp_layers = [self.user_feature_dimension + embedding_size] + mlp_layers
        # TODO Add activation functions
        self.user_mlp = torch.nn.Sequential(
            *[
                nn.Linear(mlp_layers[i], mlp_layers[i + 1])
                for i in range(0, len(mlp_layers) - 1)
            ]
        )

        # Item mlp
        mlp_layers = [self.item_feature_dimension + embedding_size] + mlp_layers
        # TODO Add activation functions
        self.item_mlp = torch.nn.Sequential(
            *[
                nn.Linear(mlp_layers[i], mlp_layers[i + 1])
                for i in range(0, len(mlp_layers) - 1)
            ]
        )
        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if data_schema["objetive"] == "binary"
            else torch.nn.MSELoss()
        )  # Only implicit feedback

        self.lr_rate = lr_rate
        self.similarity_function = similarity_function

        # Trainer
        self._trainer = PytorchLightningLiteTrainer(accelerator=accelerator)

    def sim_function(self, user_factor, item_factor):
        if self.similarity_function == "dot":
            similarity = torch.mul(user_factor, item_factor).sum(dim=1)
        elif self.similarity_function == "cosine":
            similarity = torch.cosine_similarity(user_factor, item_factor, dim=1)

        return similarity
        # TODO more funcs

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
        yhat = self.sim_function(user_factor, item_factor) + user_bias + item_bias

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
        # TODO MAKE this run under pl
        with torch.no_grad():
            torch.set_grad_enabled(False)
            self.eval()
            item_feature_dict = interactions_dataset.item_features

            # TODO OPTIMIZE THIS LOOP
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
        return item_feature_dict.keys(), item_embeddings

    def generate_user_representations(self, interactions_dataset):
        # TODO MAKE this run under pl
        with torch.no_grad():
            user_feature_dict = interactions_dataset.user_features

            # TODO OPTIMIZE THIS LOOP
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
        return user_feature_dict.keys(), user_embeddings
