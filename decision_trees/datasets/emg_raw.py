import csv
import os
from typing import Tuple, List

import numpy as np

from decision_trees.datasets.dataset_base import DatasetBase


class EMGRaw(DatasetBase):
    def __init__(self, path: str):
        self._path = path

    @staticmethod
    def _load_files(files_paths: List[str]) -> np.ndarray:
        data: List[Tuple[float, ...]] = []
        for file_path in files_paths:
            with open(file_path) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    data.append(tuple(map(float, row)))

        return np.array(data, dtype=np.float32)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        input_train_files = []
        output_train_files = []
        input_test_files = []
        output_test_files = []
        for file in os.scandir(self._path):
            file: os.DirEntry
            if 'trainvalid_i' in file.name:
                input_train_files.append(file.path)
            elif 'trainvalid_o' in file.name:
                output_train_files.append(file.path)
            elif 'test_i' in file.name:
                input_test_files.append(file.path)
            elif 'test_o' in file.name:
                output_test_files.append(file.path)

        input_train_data = self._load_files(input_train_files)
        output_train_data = self._load_files(output_train_files)
        input_test_data = self._load_files(input_test_files)
        output_test_data = self._load_files(output_test_files)

        return input_train_data, output_train_data, input_test_data, output_test_data

    @staticmethod
    def _normalise(data: np.ndarray):
        pass
