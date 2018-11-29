from typing import Tuple
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle

from decision_trees.datasets.dataset_base import DatasetBase


class MnistRaw(DatasetBase):
    def __init__(self, number_of_train_samples: int, number_of_test_samples: int):
        self._number_of_train_samples = number_of_train_samples
        self._number_of_test_samples = number_of_test_samples

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mnist = datasets.fetch_mldata('MNIST original', data_home="./../../data/datasets/MNIST/")
        # print(mnist.data.shape)
        # print(mnist.target.shape)
        # print(np.unique(mnist.target))

        # it is necessary to shuffle the data as all 0's are at the front and all 9's are at the end
        mnist.data, mnist.target = shuffle(mnist.data, mnist.target)

        train_data = self._normalise(mnist.data[:self._number_of_train_samples])
        train_target = mnist.target[:self._number_of_train_samples]
        test_data = self._normalise(
            mnist.data[self._number_of_train_samples:self._number_of_train_samples+self._number_of_test_samples]
        )
        test_target = mnist.target[
                      self._number_of_train_samples:self._number_of_train_samples+self._number_of_test_samples
                      ]

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
    # MNIST DATABASE
    # total number of samples: 70000 (each is 28x28)
    number_of_train_samples = 1000  # 65000
    number_of_test_samples = 100  # 70000 - number_of_train_samples
    # END OF PARAMETERS SETTING
    if (number_of_train_samples + number_of_test_samples) > 70000:
        print("ERROR, too many samples set!")
    #####################################

    d = MnistRaw(number_of_train_samples, number_of_test_samples)
    d.test_as_classifier(8, './../../data/vhdl/')

    assert True


def main():
    d = MnistRaw(60000, 10000)

    train_data, train_target, test_data, test_target = d.load_data()
    print(f"train_data.shape: {train_data.shape}")
    print(f"np.unique(test_target): {np.unique(test_target)}")

    d.test_as_classifier(8, './../../data/vhdl/')


if __name__ == "__main__":
    main()
