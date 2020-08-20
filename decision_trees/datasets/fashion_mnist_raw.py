import numpy as np
from typing import Tuple

from decision_trees.datasets.dataset_base import DatasetBase

import sys
sys.path.insert(0, './../../submodules/fashion-mnist/')
from utils.mnist_reader import load_mnist


class FashionMnistRaw(DatasetBase):
    def __init__(self):
        ...

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, y_train = load_mnist('./../../submodules/fashion-mnist/data/fashion', kind='train')
        X_test, y_test = load_mnist('./../../submodules/fashion-mnist/data/fashion/', kind='t10k')

        train_data = self._normalise(X_train)
        train_target = y_train
        test_data = self._normalise(X_test)
        test_target = y_test

        return train_data, train_target, test_data, test_target

    @staticmethod
    def _normalise(data: np.ndarray):
        # in case of MNIST data it is possible to just divide each data by maximum value
        # each feature is in range 0-255
        data = data / 255

        return data


def test_fashion_mnist_raw():
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # MNIST FASHION DATABASE
    # number of train samples: 60000 (each is 28x28)
    # number of test samples: 10000 (each is 28x28)
    # END OF PARAMETERS SETTING
    #####################################

    d = FashionMnistRaw()
    d.test_as_classifier(8, './../../data/vhdl/')

    assert True


def main():
    d = FashionMnistRaw()

    train_data, train_target, test_data, test_target = d.load_data()
    print(f"train_data.shape: {train_data.shape}")
    print(f"test_data.shape: {test_data.shape}")
    print(f"np.unique(test_target): {np.unique(test_target)}")

    # d.test_as_classifier(8, './../../data/vhdl/')

    # # this is the same as the code above, but on the fraction of the dataset and only for decision tree
    from decision_trees import dataset_tester
    from decision_trees.utils.constants import ClassifierType

    dataset_tester.test_dataset(
        8,
        # train_data[:60000], train_target[:60000], test_data[:10000], test_target[:10000],
        train_data, train_target, test_data, test_target,
        ClassifierType.DECISION_TREE,
        max_depth=None, number_of_classifiers=10,
        path='./../../data/vhdl/', name=d.__class__.__name__
    )

    # for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    #     dataset_tester.test_dataset(
    #         i,
    #         # train_data[:60000], train_target[:60000], test_data[:10000], test_target[:10000],
    #         train_data, train_target, test_data, test_target,
    #         ClassifierType.RANDOM_FOREST,
    #         max_depth=None, number_of_classifiers=10,
    #         path='./../../data/vhdl/', name=d.__class__.__name__
    #     )

    dataset_tester.test_dataset(
        8,
        # train_data[:60000], train_target[:60000], test_data[:10000], test_target[:10000],
        train_data, train_target, test_data, test_target,
        ClassifierType.RANDOM_FOREST,
        max_depth=None, number_of_classifiers=100,
        path='./../../data/vhdl/', name=d.__class__.__name__
    )

    # for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    #     dataset_tester.test_dataset(
    #         i,
    #         # train_data[:60000], train_target[:60000], test_data[:10000], test_target[:10000],
    #         train_data, train_target, test_data, test_target,
    #         ClassifierType.RANDOM_FOREST,
    #         max_depth=None, number_of_classifiers=100,
    #         path='./../../data/vhdl/', name=d.__class__.__name__
    #     )

    # for max_depth in [5, 10, 20, None]:
    #     for number_of_classifiers in [10, 20, 50, 100]:
    #         dataset_tester.test_dataset(
    #             8,
    #             # train_data[:60000], train_target[:60000], test_data[:10000], test_target[:10000],
    #             train_data, train_target, test_data, test_target,
    #             ClassifierType.RANDOM_FOREST,
    #             max_depth=max_depth, number_of_classifiers=number_of_classifiers,
    #             path='./../../data/vhdl/', name=d.__class__.__name__
    #         )

    # perform_gridsearch(
    #     train_data[:60000], train_target[:60000],
    #     test_data[:10000], test_target[:10000],
    #     [16, 12, 8, 6, 4, 2, 1],
    #     ClassifierType.DECISION_TREE,
    #     GridSearchType.NONE,
    #     "./../../data/gridsearch_results/",
    #     d.__class__.__name__
    # )


if __name__ == "__main__":
    main()
