from typing import Tuple

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_trees.datasets.dataset_base import DatasetBase


class BostonRaw(DatasetBase):
    def __init__(self):
        pass

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        boston = datasets.load_boston()
        # print(boston.data.shape)
        # print(boston.target.shape)
        # print(np.unique(boston.target))

        data = self._normalise(boston.data)
        train_data, test_data, train_target, test_target = train_test_split(data, boston.target, test_size=0.1,
                                                                            random_state=42)

        return train_data, train_target, test_data, test_target

    @staticmethod
    def _normalise(data: np.ndarray):
        return (data - np.min(data, 0)) / np.ptp(data, 0)


def test_boston_raw():
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # Boston House Prices DATABASE
    # total number of samples: 506 (each is 13 values)
    number_of_train_samples = 456
    number_of_test_samples = 50
    # END OF PARAMETERS SETTING
    if (number_of_train_samples + number_of_test_samples) > 506:
        print("ERROR, too much samples set!")
    #####################################

    d = BostonRaw(number_of_train_samples, number_of_test_samples)
    d.run()

    assert True


def main():
    d = BostonRaw()
    train_data, train_target, test_data, test_target = d.load_data()
    print(f'np.shape(train_data): {np.shape(train_data)}')
    print(f'np.unique(test_target): {np.unique(test_target)}')

    from decision_trees import dataset_tester

    dataset_tester.test_dataset(32,
                                train_data, train_target, test_data, test_target,
                                dataset_tester.ClassifierType.RANDOM_FOREST_REGRESSOR,
                                )


if __name__ == '__main__':
    main()
