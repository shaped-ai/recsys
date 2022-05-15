import torch
from torch import nn

from torchrecsys.models.base import BaseModel
from torchrecsys.models.utils import schema_to_featureModuleList


class ALS(BaseModel):
    def __init__(
        self,
        data_schema,
        lr_rate: float = 0.01,
        embedding_size: int = 64,
        feature_embedding_size: int = 8,
    ):
        super().__init__()
        # We will handle optimization for ALS and not PL.
        self.automatic_optimization = False

        interactions_schema = data_schema["interactions"]

        self.n_users = interactions_schema[0]
        self.n_items = interactions_schema[1]

        # User features encoding
        self.user_features = nn.ModuleList()
        self.user_feature_dimension = 0

        self.user_features, self.user_feature_dimension = schema_to_featureModuleList(
            data_schema["user_features"], feature_embedding_size
        )

        # Item features encoding
        self.item_features, self.item_feature_dimension = schema_to_featureModuleList(
            data_schema["item_features"], feature_embedding_size
        )

        aux_user_id_dimensions = self.user_feature_dimension + embedding_size
        aux_item_id_dimensions = self.item_feature_dimension + embedding_size

        max_dimension = max(aux_user_id_dimensions, aux_user_id_dimensions)
        user_id_dimensions = embedding_size + (max_dimension - aux_user_id_dimensions)
        item_id_dimensions = embedding_size + (max_dimension - aux_item_id_dimensions)

        self.user_embedding = nn.Embedding(self.n_users + 1, user_id_dimensions)
        self.item_embedding = nn.Embedding(self.n_items + 1, item_id_dimensions)

        self.user_biases = nn.Embedding(self.n_users, 1)
        self.item_biases = nn.Embedding(self.n_items, 1)

        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if data_schema["objetive"] == "binary"
            else torch.nn.MSELoss()
        )

        self.lr_rate = lr_rate

    def forward(self, interactions, context, users, items):

        user = self.user_embedding(interactions[:, 0].long())
        item = self.item_embedding(interactions[:, 1].long())

        user_features = self.encode_user(users)
        item_features = self.encode_item(items)

        user_factor = torch.cat([user, user_features], dim=1)
        item_factor = torch.cat([item, item_features], dim=1)

        pred = self.user_biases(interactions[:, 0].long()) + self.item_biases(
            interactions[:, 1].long()
        )

        pred += (user_factor * item_factor).sum(1, keepdim=True)
        return pred.squeeze()

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
        user_opt, item_opt = self.optimizers()

        ##########################
        # User factor            #
        ##########################
        yhat = self(*batch).float()
        yhat = torch.squeeze(yhat)
        ytrue = batch[0][:, 2].float()
        user_loss = self.criterion(yhat, ytrue)

        user_opt.zero_grad()
        self.manual_backward(user_loss)
        user_opt.step()

        ##########################
        # Item factor            #
        ##########################
        yhat = self(*batch).float()
        yhat = torch.squeeze(yhat)
        ytrue = batch[0][:, 2].float()
        item_loss = self.criterion(yhat, ytrue)

        item_opt.zero_grad()
        self.manual_backward(item_loss)
        item_opt.step()

        self.log(
            "train/user_step_loss",
            user_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            "train/item_step_loss",
            item_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        # NOT displaying loss on tqdm progress bar of pytorch lightning, need to fix this
        # TODO

    def validation_step(self, batch):
        yhat = self(*batch).float()
        ytrue = batch[0][:, 2].float()
        loss = self.criterion(yhat, ytrue)

        self.log(
            "validation/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False
        )

        return loss

    def configure_optimizers(self):
        user_weights = [
            self.user_embedding.weight,
            self.user_biases.weight,
        ] + [layer.weight for layer in self.user_features if hasattr(layer, "weight")]
        item_weights = [
            self.item_embedding.weight,
            self.item_biases.weight,
        ] + [layer.weight for layer in self.item_features if hasattr(layer, "weight")]

        # Need to figure out how to do the custom loss steps, when i get
        # internet
        item_optimizer = torch.optim.SGD(user_weights, self.lr_rate)
        user_optimizer = torch.optim.SGD(item_weights, self.lr_rate)
        return item_optimizer, user_optimizer
