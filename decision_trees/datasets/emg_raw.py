import csv
import os
from typing import Tuple, List

import numpy as np

from decision_trees.datasets.dataset_base import DatasetBase


class EMGRaw(DatasetBase):
    def __init__(self, path: str):
        self._path = path
        self._min_rms = float('inf')
        self._max_rms = 0.0
        self._min_zero_crossings = float('inf')
        self._max_zero_crossings = 0.0

    def _update_min_max(self, data: np.ndarray):
        rms = data[:, :8]
        min_rms = np.amin(rms)
        max_rms = np.amax(rms)

        zero_crossings = data[:, 8:]
        min_zero_crossings = np.amin(zero_crossings)
        max_zero_crossings = np.amax(zero_crossings)

        if min_rms < self._min_rms:
            self._min_rms = min_rms
        if max_rms > self._max_rms:
            self._max_rms = max_rms
        if min_zero_crossings < self._min_zero_crossings:
            self._min_zero_crossings = min_zero_crossings
        if max_zero_crossings > self._max_zero_crossings:
            self._max_zero_crossings = max_zero_crossings

    def _load_files(self, files_paths: List[str], is_output: bool) -> np.ndarray:
        data: List[Tuple[float, ...]] = []
        for file_path in files_paths:
            with open(file_path) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if is_output:
                        data.append((row.index('1'),))
                    else:
                        data.append(tuple(map(float, row)))

        data_array = np.array(data, dtype=np.float32)
        if not is_output:
            self._update_min_max(data_array)

        return data_array

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

        input_train_data = self._load_files(input_train_files, is_output=False)
        output_train_data = self._load_files(output_train_files, is_output=True)
        input_test_data = self._load_files(input_test_files, is_output=False)
        output_test_data = self._load_files(output_test_files, is_output=True)

        return input_train_data, output_train_data, input_test_data, output_test_data

    def _normalise(self, data: np.ndarray):
        data[:, :8] = (data[:, :8] - self._min_rms) / (self._max_rms - self._min_rms)
        data[:, 8:] = (data[:, 8:] - self._min_zero_crossings) / (self._max_zero_crossings - self._min_zero_crossings)

        return data
