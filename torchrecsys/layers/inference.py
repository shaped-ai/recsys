import torch

# TODO
# Batchzy the forward function so no need to fit all candidatees in mem (?)


class BruteForceLayer(torch.nn.Module):
    def __init__(self, query_model, k: int = 14):
        super().__init__()

        self.query_model = query_model
        self.k = k

    def index(self, candidates, identifiers=None):
        self._candidates = candidates

    def _compute_score(self, queries, candidates):
        scores = torch.matmul(queries, torch.transpose(candidates, 0, 1))
        return scores

    def forward(self, queries):

        if self.query_model is not None:
            queries = self.query_model(queries)

        scores = self._compute_score(queries, self._candidates)

        values, indices = torch.topk(scores, k=self.k)

        return values, indices
