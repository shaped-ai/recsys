import torch


class FeatureLayer(torch.nn.Module):
    def __init__(self, name: str, layer: torch.nn.Module, idx: int) -> None:
        super().__init__()
        self.name = name
        self.layer = layer
        self.idx = idx

    def forward(self, x):
        return self.layer(x)


class CategoricalLayer(FeatureLayer):
    def __init__(
        self, name: str, n_unique_values: int, dimensions: int, idx: int
    ) -> None:
        layer = torch.nn.Embedding(
            num_embeddings=n_unique_values + 1, embedding_dim=dimensions
        )
        super().__init__(name=name, layer=layer, idx=idx)


class NumericalLayer(FeatureLayer):
    def __init__(self, name: str, idx: int) -> None:

        super().__init__(name=name, layer=_unqueeze_fn, idx=idx)


def _unqueeze_fn(x):
    return torch.unsqueeze(x, dim=1)
