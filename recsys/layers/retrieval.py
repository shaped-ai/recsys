import torch


def retrieve_nearest_neighbors(
    candidates,
    query,
    k: int = 14,
    algo: str = "bruteforce",  # TODO ANNOY,SCANN
):
    """
    Retrieve the nearest neighbors of the query.
    """
    # TODO DATA CHECKS, shape etc

    if algo == "bruteforce":
        dist = torch.norm(candidates - query, dim=1, p=None)
        knn = dist.topk(k, largest=False)
        return knn.indices
    else:
        raise NotImplementedError("Algo not implemented")
