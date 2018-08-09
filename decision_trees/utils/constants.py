from typing import Dict
from enum import Enum, auto

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier


class ClassifierType(Enum):
    DECISION_TREE = auto()
    RANDOM_FOREST = auto()
    RANDOM_FOREST_REGRESSOR = auto()


class GridSearchType(Enum):
    SCIKIT = auto()
    PARFIT = auto()
    NONE = auto()


def get_classifier(clf_type: ClassifierType):
    if clf_type == ClassifierType.DECISION_TREE:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=None, splitter="random", random_state=42)
    elif clf_type == ClassifierType.RANDOM_FOREST:
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=3, random_state=42)
    elif clf_type == ClassifierType.RANDOM_FOREST_REGRESSOR:
        clf = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=3, random_state=42)
    else:
        raise ValueError("Unknown classifier type specified")

    return clf


def get_tuned_parameters(clf_type: ClassifierType) -> Dict:
    # TODO - min_samples_split could be a float (0.0-1.0) to tell the percentage - test it!

    # general observations:
    # for random_forest increasing the min_samples_split decreases performance, checking values above 20 is not useful
    # in general best results are obtained using min_samples_split=2 (default)

    if clf_type == ClassifierType.DECISION_TREE:
        tuned_parameters = {
            'max_depth': [10, 20, 50, 100, None],  # [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            # 'splitter': ["best", "random"],
            'min_samples_split': [2, 5, 10],  # [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            # 'min_samples_split': [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.05]
            'random_state': [42]
        }
    elif clf_type == ClassifierType.RANDOM_FOREST:
        tuned_parameters = {
            'max_depth': [10, 20, 50, 100, None],
            # criterion
            'n_estimators': [16, 32, 64, 128, 256],
            # 'max_features': ['sqrt', 'log2'],
            # 'min_samples_leaf': [1, 5, 10, 25, 50, 100, 125, 150, 175, 200],
            'min_samples_split': [2, 5, 10],  # [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100]
            # 'class_weight': [None],  # , 'balanced'],
            # 'n_jobs': [-1],
            'random_state': [42]
        }
    elif clf_type == ClassifierType.RANDOM_FOREST_REGRESSOR:
        tuned_parameters = {
            'max_depth': [10, 20, 50, 100, None],
            # criterion
            'n_estimators': [16, 32, 64, 128, 256],
            # 'max_features': ['sqrt', 'log2'],
            # 'min_samples_leaf': [1, 5, 10, 25, 50, 100, 125, 150, 175, 200],
            'min_samples_split': [2, 5, 10],  # [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100]
            # 'n_jobs': [-1],
            'random_state': [42]
        }
    else:
        raise ValueError("Unknown classifier type specified")

    return tuned_parameters
