import torch
from pytorch_lightning import Trainer

from torchrecsys.models import ALS, NCF, Bert4Rec
from torchrecsys.test.fixtures import (  # NOQA
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
    dummy_seq2seq_dataset,
    dummy_user_features,
)


def test_ncf(dummy_interaction_dataset):
    dataloader = torch.utils.data.DataLoader(dummy_interaction_dataset, batch_size=2)
    model = NCF(dummy_interaction_dataset.data_schema)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, dataloader)

    pair = torch.tensor([[1, 2]])
    context = torch.tensor([])
    user = torch.tensor([[0, 1, 0, 1]])
    item = torch.tensor([[0, 0]])

    model(pair, context, user, item)


def test_als(dummy_interaction_dataset):
    dataloader = torch.utils.data.DataLoader(dummy_interaction_dataset, batch_size=2)
    model = ALS(dummy_interaction_dataset.data_schema)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, dataloader)

    pair = torch.tensor([[1, 2]])
    context = torch.tensor([])
    user = torch.tensor([[0, 1, 0, 1]])
    item = torch.tensor([[0, 0]])

    model(pair, context, user, item)


def test_bert4rec(dummy_seq2seq_dataset):
    dataloader = torch.utils.data.DataLoader(dummy_seq2seq_dataset, batch_size=2)
    model = Bert4Rec(dummy_seq2seq_dataset.data_schema)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, dataloader)

    sequence = torch.zeros(
        2, dummy_seq2seq_dataset.data_schema["max_length"], dtype=torch.long
    )
    item = torch.zeros(
        2, dummy_seq2seq_dataset.data_schema["max_length"], 2, dtype=torch.long
    )

    model(sequence, item)
