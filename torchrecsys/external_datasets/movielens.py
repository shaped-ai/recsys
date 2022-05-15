import io
import os
import zipfile
from typing import Tuple

import pandas as pd
import requests

from torchrecsys.external_datasets.base import DEFAULT_ROOT_DIR, BaseDataset

# download_url
ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_10M_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
ML_25M_URL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"


class Movielens_1M(BaseDataset):
    """Movielens 100k Dataset."""

    def __init__(
        self,
        dataset_name="ml-1m",
        data_url: str = ML_1M_URL,
        dataset_dir: str = DEFAULT_ROOT_DIR,
    ):
        """Init Movielens_100k Class."""
        super().__init__(
            dataset_name=dataset_name, data_url=data_url, dataset_dir=dataset_dir
        )

    def preprocess(self):
        """Preprocess the dataset."""

        # download and extract the dataset
        r = requests.get(self.data_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(self.src_path)

    # not happy with current results but can work it out later
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ """
        self.dataset_path = os.path.join(self.src_path, "ml-1m")
        if not os.path.exists(self.dataset_path):
            self.preprocess()

        unames = ["user_id", "gender", "age", "occupation", "zip"]
        users_path = os.path.join(self.dataset_path, "users.dat")
        users = pd.read_table(
            users_path, sep="::", header=None, names=unames, engine="python"
        )

        rnames = ["user_id", "movie_id", "rating", "timestamp"]
        ratings_path = os.path.join(self.dataset_path, "ratings.dat")
        ratings = pd.read_table(
            ratings_path, sep="::", header=None, names=rnames, engine="python"
        )
        # Movie information
        mnames = ["movie_id", "title", "genres"]
        movies_path = os.path.join(self.dataset_path, "movies.dat")
        movies = pd.read_table(
            movies_path,
            sep="::",
            header=None,
            names=mnames,
            engine="python",
            encoding="ISO-8859-1",
        )

        return ratings, users, movies
