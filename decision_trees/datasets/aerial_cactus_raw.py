from typing import Tuple
import numpy as np
import os
import cv2
import random
import pandas

from decision_trees.datasets.dataset_base import DatasetBase


class AerialCactusRaw(DatasetBase):
    def __init__(
            self, path: str,
            number_of_train_samples: int, number_of_test_samples: int
    ):
        random.seed(42)
        self._path = path
        self._number_of_train_samples = number_of_train_samples
        self._number_of_test_samples = number_of_test_samples

        # NOTE(MF): does not work, as it gives a string as bte array that can not be easily used to index
        # labels2 = dict(np.genfromtxt(
        #     os.path.join(self._path, 'train.csv'),
        #     delimiter=',', skip_header=1,
        #     dtype=[('file_name', '<S50'), ('flag', 'i1')]
        # ))
        # print(len(labels2))
        # print(labels2)

        labels = dict(pandas.read_csv(os.path.join(self._path, 'train.csv')).as_matrix())
        # print(len(labels))
        # print(labels)

        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(os.path.join(self._path, 'train')):
            for file in f:
                if '.jpg' in file:
                    files.append(os.path.join(r, file))

        print(len(files))
        data = {}
        for f in files:
            data[os.path.basename(f)] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        print(len(data))

        keys = list(data.keys())
        random.shuffle(keys)

        self._train_data = []
        self._train_target = []
        self._test_data = []
        self._test_target = []
        for key in keys[:self._number_of_train_samples]:
            # print(data[key].shape)
            # data has to be flatten (8x8 image -> 64x1 matrix)
            d = data[key].flatten()
            # print(d.shape)
            # print(d)
            d = self._normalise(d)
            # print(d)
            self._train_data.append(d)
            self._train_target.append(labels[key])
        for key in keys[self._number_of_train_samples:self._number_of_train_samples + self._number_of_train_samples]:
            d = self._normalise(data[key].flatten())
            self._test_data.append(d)
            self._test_target.append(labels[key])

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self._train_data), np.asarray(self._train_target), \
               np.asarray(self._test_data), np.asarray(self._test_target)

    @staticmethod
    def _normalise(data: np.ndarray):
        # in case of digits data it is possible to just divide each data by maximum value
        # each feature is in range 0-16
        data = data / 256

        return data


def main():
    # d = AerialCactusRaw('./../../data/datasets/aerial-cactus-identification/', 15000, 2500)
    d = AerialCactusRaw('./../../data/datasets/aerial-cactus-identification/', 15000, 250)

    # train_data, train_target, test_data, test_target = d.load_data()
    # print(train_data[0])
    # print(train_target[0])
    #
    # print(test_data[10])
    # print(test_target[10])

    for i in range(1, 9):
        d.test_as_classifier(i, './../../data/vhdl/')


if __name__ == '__main__':
    main()
