import io
import os
import zipfile
from typing import Tuple

import pandas as pd
import requests

from torchrecsys.external_datasets.base import DEFAULT_ROOT_DIR, BaseDataset

# Download links
NETFLIX_DRIVE_URL = "https://github.com/jiwidi/netflix_price/raw/master/netflix.zip"


class Netflix(BaseDataset):
    """Netflix"""

    def __init__(
        self,
        dataset_dir: str = DEFAULT_ROOT_DIR,
        data_url: str = NETFLIX_DRIVE_URL,
    ):
        """Init Netflix dataset class."""
        super().__init__(
            dataset_name="netflix", data_url=data_url, dataset_dir=dataset_dir
        )

    def download_dataset(self, destination_path):
        """Download the dataset."""
        r = requests.get(self.data_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(destination_path)

    def load(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        self.dataset_path = os.path.join(self.src_path, self.dataset_name)
        if not os.path.exists(self.dataset_path):
            self.download_dataset(self.dataset_path)

        ratings = pd.read_parquet(
            os.path.join(self.dataset_path, "netflix/ratings.parquet")
        )

        movies = pd.read_parquet(
            os.path.join(self.dataset_path, "netflix/movie_titles.parquet")
        )

        users = pd.Series(ratings["User"].unique(), name="User")
        return ratings, users, movies
