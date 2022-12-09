from recsys.external_datasets import Movielens_1M
from recsys.datasets import InteractionsDataset, Seq2SeqDataset

def test_movielens_1m():
    data = Movielens_1M()
    ratings, users, movies = data.load()
    assert ratings.shape == (1000209, 4)
    assert users.shape == (6040, 5)
    assert movies.shape == (3883, 3)

    # Build a dataset.
    # Todo fn to process data into right format then test datasets
    # dataset = InteractionsDataset(
    #     interactions=ratings,
    #     user_features=users,
    #     item_features=movies,
    #     interaction_id="rating",
    #     user_id="user_id",
    #     item_id="movie_id",
    # )
