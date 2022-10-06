import torch

from torchrecsys.layers.retrieval import retrieve_nearest_neighbors


def test_retrieve_nearest_neighbors():
    candidates = [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1],
        [7, 7, 7, 7, 7, 7],
        [500, 500, 500, 500, 500, 500],
        [3, 3, 3, 3, 3, 3],
    ]

    aux = retrieve_nearest_neighbors(candidates=candidates, query=candidates[0])
    assert torch.equal(aux, torch.tensor([0, 1, 4, 2, 3], dtype=torch.int32))

    aux = retrieve_nearest_neighbors(candidates=candidates, query=candidates[1])
    assert torch.equal(aux, torch.tensor([1, 0, 4, 2, 3], dtype=torch.int32))
