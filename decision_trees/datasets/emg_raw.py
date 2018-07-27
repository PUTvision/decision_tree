from typing import Tuple, List
import csv
import os
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

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        input_train_data = self._normalise(input_train_data)
        input_test_data = self._normalise(input_test_data)

        return input_train_data, output_train_data, input_test_data, output_test_data

    def _normalise(self, data: np.ndarray):
        data[:, :8] = (data[:, :8] - self._min_rms) / (self._max_rms - self._min_rms)
        data[:, 8:] = (data[:, 8:] - self._min_zero_crossings) / (self._max_zero_crossings - self._min_zero_crossings)

        return data


if __name__ == "__main__":
    d = EMGRaw("./../../data/EMG/")

    train_data, train_target, test_data, test_target = d.load_data()

    print(f"train_data.shape: {train_data.shape}")
    print(f"test_data.shape: {test_data.shape}")
    print(f"np.unique(train_target): {np.unique(train_target)}")
    print(f"np.unique(test_target): {np.unique(test_target)}")

    from decision_trees import dataset_tester

    dataset_tester.perform_gridsearch(train_data[:19000], train_target[:19000],
                                      test_data[:10000], test_target[:10000],
                                      10 - 1,
                                      clf_type=dataset_tester.ClassifierType.RANDOM_FOREST,
                                      gridsearch_type=dataset_tester.GridSearchType.NONE
                                      )

    # dataset_tester.test_dataset(8,
    #                             train_data[:19000], train_target[:19000],
    #                             test_data[:10000], test_target[:10000],
    #                             dataset_tester.ClassifierType.RANDOM_FOREST,
    #                             )
