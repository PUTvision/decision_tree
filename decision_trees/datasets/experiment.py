from typing import List

from decision_trees.datasets.dataset_base import DatasetBase
from decision_trees.datasets.mnist_raw import MnistRaw
from decision_trees.datasets.fashion_mnist_raw import FashionMnistRaw
from decision_trees.datasets.emg_raw import EMGRaw
from decision_trees.datasets.terrain import Terrain
from decision_trees.datasets.boston_house_prices_raw import BostonRaw

from decision_trees.gridsearch import perform_gridsearch
from decision_trees.utils.constants import ClassifierType, GridSearchType


def run_classification_database(
        d: DatasetBase,
        bit_width_to_test: List[int],
        path_with_gridsearch_results: str,
):
    try:
        train_data, train_target, test_data, test_target = d.load_data()

        perform_gridsearch(
            train_data, train_target, test_data, test_target,
            bit_width_to_test,
            ClassifierType.DECISION_TREE,
            GridSearchType.SCIKIT,
            path_with_gridsearch_results,
            d.__class__.__name__
        )

        perform_gridsearch(
            train_data, train_target, test_data, test_target,
            bit_width_to_test,
            ClassifierType.RANDOM_FOREST,
            GridSearchType.SCIKIT,
            path_with_gridsearch_results,
            d.__class__.__name__
        )
    except Exception as e:
        print(e)


def run_regression_database(
        d: BostonRaw,
        bit_width_to_test: List[int],
        path_with_gridsearch_results: str,
):
    try:
        train_data, train_target, test_data, test_target = d.load_data()

        perform_gridsearch(
            train_data, train_target, test_data, test_target,
            bit_width_to_test,
            ClassifierType.RANDOM_FOREST_REGRESSOR,
            GridSearchType.SCIKIT,
            path_with_gridsearch_results,
            d.__class__.__name__
        )
    except Exception as e:
        print(e)


def experiment():
    path_with_gridsearch_results = './../../data/gridsearch_results/'
    path_with_vhdl_results = './../../data/vhdl/'

    run_classification_database(
        EMGRaw('./../../data/datasets/EMG/'),
        [16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1],
        path_with_gridsearch_results
    )
    run_classification_database(
        Terrain('./../../data/datasets/terrain_data/'),
        [16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1],
        path_with_gridsearch_results
    )

    run_regression_database(
        BostonRaw(),
        [16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1],
        path_with_gridsearch_results
    )

    # these two take most time
    run_classification_database(
        MnistRaw(65000, 5000),
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        path_with_gridsearch_results,
    )
    run_classification_database(
        FashionMnistRaw(),
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        path_with_gridsearch_results
    )


if __name__ == '__main__':
    experiment()
