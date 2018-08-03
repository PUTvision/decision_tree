from enum import Enum, auto


class ClassifierType(Enum):
    DECISION_TREE = auto()
    RANDOM_FOREST = auto()
    RANDOM_FOREST_REGRESSOR = auto()


class GridSearchType(Enum):
    SCIKIT = auto()
    PARFIT = auto()
    NONE = auto()
