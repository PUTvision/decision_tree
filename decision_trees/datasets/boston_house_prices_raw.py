from typing import Tuple
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle

from decision_trees.datasets.dataset_base import DatasetBase


class BostonRaw(DatasetBase):
    def __init__(self, number_of_train_samples: int, number_of_test_samples: int):
        self._number_of_train_samples = number_of_train_samples
        self._number_of_test_samples = number_of_test_samples

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        boston = datasets.load_boston()
        # print(boston.data.shape)
        # print(boston.target.shape)
        # print(np.unique(boston.target))

        # it is necessary to shuffle the data as all 0's are at the front and all 9's are at the end
        boston.data, boston.target = shuffle(boston.data, boston.target)

        train_data = boston.data[:self._number_of_train_samples]
        train_target = boston.target[:self._number_of_train_samples]
        test_data = boston.data[self._number_of_train_samples:self._number_of_train_samples+self._number_of_test_samples]
        test_target = boston.target[self._number_of_train_samples:self._number_of_train_samples+self._number_of_test_samples]

        # TODO(MF): insert normalisation routine here

        return train_data, train_target, test_data, test_target

    @staticmethod
    def _normalise(data: np.ndarray):
        # in case of MNIST data it is possible to just divide each data by maximum value
        # each feature is in range 0-255
        # TODO(MF): add normalisation
        data = data / 255

        return data


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


if __name__ == "__main__":
    d = BostonRaw(400, 100)

    train_data, train_target, test_data, test_target = d.load_data()

    print(f"train_data.shape: {train_data.shape}")

    from sklearn.ensemble import RandomForestRegressor

    clf = RandomForestRegressor(n_estimators=10, max_depth=None, n_jobs=3, random_state=42)
    clf.fit(train_data, train_target)
    test_predicted = clf.predict(test_data)

    diffs = test_target - test_predicted
    for diff, expected, predicted in zip(diffs, test_target, test_predicted):
        print(f"expected: {expected}, predicted: {predicted}, diff: {diff:{1}.{3}}, relative error: {diff/expected:{1}.{3}}")

    # dataset_tester.test_dataset(40,
    #                             train_data, train_target, test_data, test_target,
    #                             dataset_tester.ClassifierType.random_forest_regressor,
    #                             )
