import abc
from typing import Tuple

import numpy as np

from decision_trees import dataset_tester


class DatasetBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _normalise(data: np.ndarray) -> np.ndarray:
        pass

    def run(self):
        train_data, train_target, test_data, test_target = self.load_data()

        dataset_tester.test_dataset(8,
                                    train_data, train_target, test_data, test_target,
                                    dataset_tester.ClassifierType.DECISION_TREE,
                                    )

    def run_grid_search(self):
        train_data, train_target, test_data, test_target = self.load_data()

        dataset_tester.grid_search(train_data, train_target,
                                   test_data, test_target,
                                   dataset_tester.ClassifierType.DECISION_TREE
                                   )
