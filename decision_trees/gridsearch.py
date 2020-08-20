from typing import List

import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier

# TODO(MF): original parfit did not work correctly with our data
# from parfit.parfit import bestFit, plotScores
from decision_trees.own_parfit.parfit import bestFit
from decision_trees.utils.constants import ClassifierType
from decision_trees.utils.constants import GridSearchType
from decision_trees.utils.constants import get_classifier, get_tuned_parameters
from decision_trees.utils.convert_to_fixed_point import quantize_data


def perform_gridsearch(train_data: np.ndarray, train_target: np.ndarray,
                       test_data: np.ndarray, test_target: np.ndarray,
                       number_of_bits_per_feature_to_test: List[int],
                       clf_type: ClassifierType,
                       gridsearch_type: GridSearchType,
                       path: str,
                       name: str
                       ):
    filename_with_path = path + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') \
                         + '_' + name + '_' + clf_type.name + '_gridsearch_results.txt'

    # first train on the non-quantized data
    if gridsearch_type == GridSearchType.SCIKIT:
        best_model, best_score = _scikit_gridsearch(train_data, train_target, test_data, test_target, clf_type)
    elif gridsearch_type == GridSearchType.PARFIT:
        best_model, best_score = _parfit_gridsearch(train_data, train_target, test_data, test_target, clf_type, False)
    elif gridsearch_type == GridSearchType.NONE:
        best_model, best_score = _none_gridsearch(train_data, train_target, test_data, test_target, clf_type)
    else:
        raise ValueError('Requested GridSearchType is not available')

    print('No quantization - full resolution')
    with open(filename_with_path, 'a') as f:
        print('No quantization - full resolution', file=f)
    _save_score_and_model_to_file(best_score, best_model, filename_with_path)

    # repeat on quantized data with different number of bits
    for bit_width in number_of_bits_per_feature_to_test:
        train_data_quantized, test_data_quantized = quantize_data(
            train_data, test_data, bit_width,
            flag_save_details_to_file=False, path='./../../data/'
        )

        if gridsearch_type == GridSearchType.SCIKIT:
            best_model, best_score = _scikit_gridsearch(
                train_data_quantized, train_target,
                test_data, test_target,
                clf_type
            )
        elif gridsearch_type == GridSearchType.PARFIT:
            best_model, best_score = _parfit_gridsearch(
                train_data_quantized, train_target,
                test_data_quantized, test_target,
                clf_type, show_plot=False
            )
        elif gridsearch_type == GridSearchType.NONE:
            best_model, best_score = _none_gridsearch(
                train_data_quantized, train_target,
                test_data, test_target,
                clf_type
            )
        else:
            raise ValueError('Requested GridSearchType is not available')

        print(f'number of bits: {bit_width}')
        with open(filename_with_path, 'a') as f:
            print(f'number of bits: {bit_width}', file=f)
        _save_score_and_model_to_file(best_score, best_model, filename_with_path)


def _save_score_and_model_to_file(score, model, filename: str):
    print(f"f1: {score:{1}.{5}}: {model}")
    with open(filename, "a") as f:
        print(f"f1: {score:{1}.{5}}: {model}", file=f)


# this should use the whole data available to find best parameters. So pass both train and test data, find parameters
# and then retrain with found parameters on train data
def _scikit_gridsearch(
        train_data: np.ndarray, train_target: np.ndarray,
        test_data: np.ndarray, test_target: np.ndarray,
        clf_type: ClassifierType
):
    # perform grid search to find best parameters
    scores = ['neg_mean_squared_error'] if clf_type == ClassifierType.RANDOM_FOREST_REGRESSOR else ['f1_weighted']
    # alternatives: http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values

    tuned_parameters = get_tuned_parameters(clf_type)

    # for score in scores:
    score = scores[0]

    # print("# Tuning hyper-parameters for %s" % score)
    # print()

    # TODO: important note - this does not use test data to evaluate, instead it probably splits the train data
    # internally, which means that the final score will be calculated on this data and is different than the one
    # calculated on test data

    data = np.concatenate((train_data, test_data))
    target = np.concatenate((train_target, test_target))

    labels = np.full((len(data),), -1, dtype=np.int8)
    labels[len(train_data):] = 1
    cv = PredefinedSplit(labels)

    if clf_type == ClassifierType.DECISION_TREE:
        clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=cv, scoring=score, n_jobs=3)
    elif clf_type == ClassifierType.RANDOM_FOREST:
        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=cv, scoring=score, n_jobs=3)
    elif clf_type == ClassifierType.RANDOM_FOREST_REGRESSOR:
        clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=cv, scoring=score, n_jobs=3)
    else:
        raise ValueError('Unknown classifier type specified')

    clf = clf.fit(data, target)

    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set:")
    # for mean, std, params in zip(
    #         clf.cv_results_['mean_test_score'],
    #         clf.cv_results_['std_test_score'],
    #         clf.cv_results_['params']
    # ):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()

    return clf.best_params_, clf.best_score_


def _none_gridsearch(
        train_data: np.ndarray, train_target: np.ndarray,
        test_data: np.ndarray, test_target: np.ndarray,
        clf_type: ClassifierType
):
    clf = get_classifier(clf_type)

    clf = clf.fit(train_data, train_target)

    return clf.get_params(), clf.score(test_data, test_target)


# TODO(MF): check parfit module for parameters search
# https://medium.com/mlreview/parfit-hyper-parameter-optimization-77253e7e175e
def _parfit_gridsearch(
        train_data: np.ndarray, train_target: np.ndarray,
        test_data: np.ndarray, test_target: np.ndarray,
        clf_type: ClassifierType,
        show_plot: bool
):
    grid = get_tuned_parameters(clf_type)

    if clf_type == ClassifierType.DECISION_TREE:
        model = DecisionTreeClassifier
    elif clf_type == ClassifierType.RANDOM_FOREST:
        model = RandomForestClassifier
    else:
        raise ValueError("Unknown classifier type specified")

    from sklearn import metrics
    best_model, best_score, all_models, all_scores = bestFit(model, ParameterGrid(grid),
                                                             train_data, train_target, test_data, test_target,
                                                             predictType='predict',
                                                             metric=metrics.f1_score, bestScore='max',
                                                             scoreLabel='f1_weighted', showPlot=show_plot)

    return best_model, best_score
