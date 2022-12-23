from typing import List

import torch
from torch import nn

from recsys.models.base import BaseModel
from recsys.models.trainers import PytorchLightningLiteTrainer
from recsys.models.utils import align_item_user_dimensions, schema_to_featureModuleList


class NCF(nn.Module, BaseModel):
    def __init__(
        self,
        data_schema,
        feature_embedding_size: int = 8,
        id_embedding_size: int = 8,
        mlp_layers: List[int] = [512, 256],  # NOQA B006
        accelerator="cpu",
    ):
        super().__init__()

        # Find embedding size for both user and item IDS so that they final vector with
        # the features is the same size
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

        user_item_combined_dimension = (
            self.user_feature_dimension + self.item_feature_dimension
        )
        mlp_layers = [user_item_combined_dimension] + mlp_layers
        self.mlp = torch.nn.Sequential()
        for i in range(0, len(mlp_layers) - 1):
            self.mlp.add_module(
                f"linear_{i}",
                nn.Linear(mlp_layers[i], mlp_layers[i + 1]),
            )
            self.mlp.add_module(f"relu_{i}", nn.ReLU())

        gmf_output_dimension = self.user_feature_dimension
        self.final_linear = torch.nn.Sequential(
            nn.Linear(mlp_layers[-1] + gmf_output_dimension, 1),
            nn.Sigmoid(),
        )
        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if data_schema["objetive"] == "binary"
            else torch.nn.MSELoss()
        )

        # Trainer
        self._trainer = PytorchLightningLiteTrainer(accelerator=accelerator)

    def forward(self, users, items):
        user_features = self.encode_user(users)
        item_features = self.encode_item(items)
        return self.score(user_features, item_features)

    def score(self, user_vector, item_vector):
        user_vector = torch.as_tensor(user_vector)
        item_vector = torch.as_tensor(item_vector)
        mlp_output = self.mlp(torch.cat([user_vector, item_vector], dim=1))
        gmf_output = user_vector * item_vector

        x = self.final_linear(torch.cat([gmf_output, mlp_output], dim=1))
        x = torch.squeeze(x)
        return x

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

        return torch.stack(r) if len(r) > 0 else torch.tensor(r)

    def encode_user(self, user):
        r = []
        for _idx, feature in enumerate(self.user_features):
            feature_representation = feature(user[:, feature.idx])
            r.append(feature_representation)
        r = torch.cat(r, dim=1).float() if r else None  # Concatenate all features
        return r

    def encode_item(self, item):
        r = []
        for _idx, feature in enumerate(self.item_features):
            feature_representation = feature(item[:, feature.idx])
            r.append(feature_representation)
        r = torch.cat(r, dim=1).float() if r else None  # Concatenate all features
        return r

    def training_step(self, batch):
        interactions, users, items = batch

        yhat = self(users, items).float()
        yhat = torch.squeeze(yhat)

        ytrue = interactions[:, 2].float()

        loss = self.criterion(yhat, ytrue)
        return loss

    def validation_step(self, batch):
        yhat = self(*batch).float()
        ytrue = batch[0][:, 2].float()
        loss = self.criterion(yhat, ytrue)
        return loss

    def configure_optimizers(self, lr_rate):
        optimizer = torch.optim.AdamW(self.parameters(), lr_rate)
        return optimizer
