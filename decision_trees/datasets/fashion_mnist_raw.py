import numpy as np
from sklearn.utils import shuffle
from typing import Tuple

from decision_trees.datasets.dataset_base import DatasetBase

import sys
sys.path.insert(0, './../../submodules/fashion-mnist/')
from utils.mnist_reader import load_mnist


class FashionMnistRaw(DatasetBase):
    def __init__(self):
        ...

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, y_train = load_mnist('./../../submodules/fashion-mnist/data/fashion', kind='train')
        X_test, y_test = load_mnist('./../../submodules/fashion-mnist/data/fashion/', kind='t10k')

        train_data = X_train
        train_target = y_train
        test_data = X_test
        test_target = y_test

        return train_data, train_target, test_data, test_target

    @staticmethod
    def _normalise(data: np.ndarray):
        # in case of MNIST data it is possible to just divide each data by maximum value
        # each feature is in range 0-255
        data = data / 255

        return data


def test_mnist_raw():
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # MNIST FASHION DATABASE
    # number of train samples: 60000 (each is 28x28)
    # number of test samples: 10000 (each is 28x28)
    # END OF PARAMETERS SETTING
    #####################################

    d = FashionMnistRaw()
    d.run()

    assert True


if __name__ == "__main__":
    d = FashionMnistRaw()

    train_data, train_target, test_data, test_target = d._load_data()

    print(f"train_data.shape: {train_data.shape}")
    print(f"np.unique(test_target): {np.unique(test_target)}")

    train_data = d._normalise(train_data)
    test_data = d._normalise(test_data)

    from decision_trees import dataset_tester

    dataset_tester.perform_gridsearch(train_data[:60000], train_target[:60000],
                                      test_data[:10000], test_target[:10000],
                                      10 - 1,
                                      dataset_tester.ClassifierType.DECISION_TREE,
                                      dataset_tester.GridSearchType.NONE
                                      )

    # dataset_tester.test_dataset(4,
    #                             train_data[:60000], train_target[:60000], test_data[:10000], test_target[:10000],
    #                             dataset_tester.ClassifierType.DECISION_TREE,
    #                             )
