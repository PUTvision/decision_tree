from typing import Tuple
import pickle
import numpy as np

from decision_trees.datasets.dataset_base import DatasetBase


class InriaHoG(DatasetBase):
    def __init__(self, data_filename: str, nr_pos_train: int, nr_pos_test: int, nr_neg_train: int, nr_neg_test: int):
        self._data_filename = data_filename
        self._nr_pos_train = nr_pos_train
        self._nr_pos_test = nr_pos_test
        self._nr_neg_train = nr_neg_train
        self._nr_neg_test = nr_neg_test

    def load_data(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # prepare the training data
        with open("..\\data\\positive_train_" + self._data_filename + ".pickle", "rb") as f:
            train_data_positive = pickle.load(f)

        with open("..\\data\\positive_test_" + self._data_filename + ".pickle", "rb") as f:
            test_data_positive = pickle.load(f)

        with open("..\\data\\negative_train_" + self._data_filename + ".pickle", "rb") as f:
            train_data_negative = pickle.load(f)

        with open("..\\data\\negative_test_" + self._data_filename + ".pickle", "rb") as f:
            test_data_negative = pickle.load(f)

        train_data = train_data_positive[0:self._nr_pos_train] + train_data_negative[0:self._nr_neg_train]
        train_target = [1] * self._nr_pos_train + [0] * self._nr_neg_train

        test_data = test_data_positive[0:self._nr_pos_test] + test_data_negative[0:self._nr_neg_test]
        test_target = [1] * self._nr_pos_test + [0] * self._nr_neg_test

        return train_data, train_target, test_data, test_target

    @staticmethod
    def _normalise(data: np.ndarray):
        # TODO - check how the data should be normalised
        raise NotImplementedError("datasets should implement this!")

        return data


def test_inria_hog():
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # INRIA DATABASE FOR HOG (64x128)
    # total number of positive samples: 1126, but only 1100 can be used here (900 for samples, 200 for tests)
    number_of_positive_samples = 200#900
    number_of_positive_tests = 50#200
    # total number of negative train samples: 1218, but only 1200 can be used here
    number_of_negative_samples = 400#1200
    # total number of negative test samples: 453, , but only 400 can be used here
    number_of_negative_tests = 50#400
    # END OF PARAMETERS SETTING
    #####################################

    d = InriaHoG("samples",
                 # use the following file if modifed (own HoG should be used)
                 # "samples_modified",
                 number_of_positive_samples,
                 number_of_positive_tests,
                 number_of_negative_samples,
                 number_of_negative_tests
                 )
    d.run()

    assert True


if __name__ == "__main__":
    test_inria_hog()
