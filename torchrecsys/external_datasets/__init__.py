from .base import BaseDataset
from .movielens import Movielens_1M, Movielens_25M
from .netflix import Netflix

__all__ = ["BaseDataset", "Movielens_1M", "Movielens_25M", "Netflix"]
