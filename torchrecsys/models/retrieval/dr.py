from typing import List

import torch
from torch import nn

from torchrecsys.models.base import BaseModel
from torchrecsys.models.utils import schema_to_featureModuleList


class DeepRetriever(BaseModel):
    def __init__(
        self,
        data_schema,
        lr_rate: float = 0.01,
        embedding_size: int = 64,
        feature_embedding_size: int = 8,
        mlp_layers: List[int] = [512, 256],  # NOQA B006
        similarity_function: str = "dot",
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
        self.criterion = torch.nn.BCEWithLogitsLoss()  # Only implicit feedback

        self.lr_rate = lr_rate
        self.similarity_function = similarity_function

    def sim_function(self, user_factor, item_factor):
        if self.similarity_function == "dot":
            similarity = torch.mul(user_factor, item_factor).sum(dim=1)
        elif self.similarity_function == "cosine":
            similarity = torch.cosine_similarity(user_factor, item_factor, dim=1)

        return similarity
        # TODO more funcs

    def forward(self, interactions, context, users, items):
        user = self.user_embedding(interactions[:, 0].long())
        item = self.item_embedding(interactions[:, 1].long())

        user_features = self.encode_user(users)
        item_features = self.encode_item(items)

        user_factor = torch.cat([user, user_features], dim=1)
        item_factor = torch.cat([item, item_features], dim=1)

        user_factor = self.user_mlp(user_factor)
        item_factor = self.item_mlp(item_factor)

        yhat = self.sim_function(user_factor, item_factor)

        return torch.sigmoid(yhat)

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

    def training_step(self, batch):
        interactions, context, users, items = batch

        yhat = self(interactions.long(), context, users, items).float()
        ytrue = batch[0][:, 2].float()

        loss = self.criterion(yhat, ytrue)

        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch):
        yhat = self(*batch).float()
        ytrue = batch[0][:, 2].float()
        loss = self.criterion(yhat, ytrue)

        self.log(
            "validation/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr_rate)

        return optimizer
