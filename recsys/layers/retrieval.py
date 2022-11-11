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
        candidates = torch.as_tensor(candidates, dtype=torch.float64)
        query = torch.as_tensor(query, dtype=torch.float64)
        dist = torch.norm(candidates - query, dim=1, p=None)
        k = min(k, len(candidates))
        knn = dist.topk(k, largest=False)
        return knn.indices.int()
    else:
        raise NotImplementedError("Algo not implemented")
