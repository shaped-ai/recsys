import os
from abc import ABC

DEFAULT_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


class BaseDataset(ABC):
    def __init__(
        self,
        dataset_name: str,
        data_url: str,
        dataset_dir: str = DEFAULT_ROOT_DIR,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.data_url = data_url

        # create the dataset directory
        self.dataset_dir = os.path.join(self.dataset_dir, "external_datasets")
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        # create the src directory
        self.src_path = os.path.join(self.dataset_dir, "raw_files")
        if not os.path.exists(self.src_path):
            os.mkdir(self.src_path)
