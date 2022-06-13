import os
from typing import Tuple
import pandas as pd
from torchrecsys.external_datasets.base import DEFAULT_ROOT_DIR, BaseDataset


class Netflix(BaseDataset):
    """Netflix"""

    def __init__(
        self,
        dataset_name="netflix",
        data_url: str = "https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data",
        dataset_dir: str = DEFAULT_ROOT_DIR,
        kaggle_username: str = "xxxx",
        kaggle_api: str = "xxxxxxx"
    ):
        """Init Netflix Class."""
        super().__init__(
            dataset_name=dataset_name, data_url=data_url, dataset_dir=dataset_dir
        )
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_api

    def download_dataset(self):
        """Download the dataset, requires Kaggle credentials"""
        import kaggle
        
        print("Downloading dataset")
        kaggle.api.dataset_download_files('netflix-inc/netflix-prize-data', path=self.dataset_path, unzip=True)
        self.preprocess()

    def preprocess(self):
        """Preprocess the dataset."""

        
        data = open(os.path.join(self.dataset_path, "data.csv"), mode="w")

        files = [
            os.path.join(self.dataset_path, "combined_data_1.txt"),
            os.path.join(self.dataset_path, "combined_data_2.txt"),
            os.path.join(self.dataset_path, "combined_data_3.txt"),
            os.path.join(self.dataset_path, "combined_data_4.txt"),
        ]

        for file in files:
            print("Opening file: {}".format(file))
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(":"):
                        movie_id = line.replace(":", "")
                    else:
                        data.write(movie_id + "," + line)
                        data.write("\n")
        data.close()

    def load(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        
        self.dataset_path = os.path.join(self.src_path, "netflix")
        if not os.path.exists(self.dataset_path):
            self.download_dataset()
        elif not os.path.isfile(os.path.join(self.dataset_path, "data.csv")):
            self.preprocess()
            
        ratings = pd.read_csv(
            os.path.join(self.dataset_path, "data.csv"),
            names=["movie_id", "user_id", "rating", "date"],
            parse_dates=["date"],
            encoding="utf8",
            engine="python",
        )
        
        movies = pd.read_csv(
            os.path.join(self.dataset_path, "data.csv"), 
            names=["movie_id", "release_year", "title"],
            encoding = "ISO-8859-1"
        )

        users = pd.Series(ratings["user_id"].unique())
        return ratings, users, movies
