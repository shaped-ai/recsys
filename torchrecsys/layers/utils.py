import torch


def compute_similarity(x, y, mode="dot"):
    if mode == "dot":
        similarity = torch.mul(x, y).sum(dim=1)
    elif mode == "cosine":
        similarity = torch.cosine_similarity(x, y, dim=1)

    return similarity