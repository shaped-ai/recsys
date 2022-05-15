from torchrecsys.external_datasets import Movielens_1M


def test_movielens_1m():
    data = Movielens_1M()
    ratings, users, movies = data.load()
