import abc
import numpy as np
import typing


import dataset_tester


class DatasetBase:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _load_data(self) -> typing.Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return

    @staticmethod
    @abc.abstractmethod
    def _normalise(data: np.ndarray):
        return

    def run(self):
        train_data, train_target, test_data, test_target = self._load_data()

        train_data = self._normalise(train_data)
        test_data = self._normalise(test_data)

        # dataset_tester.test_dataset(4,
        #                             train_data, train_target, test_data, test_target,
        #                             dataset_tester.ClassifierType.decision_tree
        #                             )
        dataset_tester.grid_search(np.concatenate((train_data, test_data)),
                                   np.concatenate((train_target, test_target)),
                                   dataset_tester.ClassifierType.decision_tree
                                   )
