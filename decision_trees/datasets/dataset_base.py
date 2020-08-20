import abc
from typing import Tuple

import numpy as np

from decision_trees.dataset_tester import test_dataset
from decision_trees.utils.constants import ClassifierType, GridSearchType


class DatasetBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _normalise(data: np.ndarray) -> np.ndarray:
        pass

    def test_as_classifier(self, number_of_bits_per_feature: int, path: str):
        train_data, train_target, test_data, test_target = self.load_data()

        print('Testing decision tree classifier')
        test_dataset(
            number_of_bits_per_feature,
            train_data, train_target, test_data, test_target,
            ClassifierType.DECISION_TREE,
            None,
            None,
            path,
            self.__class__.__name__
        )

        print('Testing random forest classifier')
        test_dataset(
            number_of_bits_per_feature,
            train_data, train_target, test_data, test_target,
            ClassifierType.RANDOM_FOREST,
            None,
            100,
            path,
            self.__class__.__name__
        )
