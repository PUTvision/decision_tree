import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from decision_trees.utils.constants import ClassifierType
from decision_trees.vhdl_generators.tree import Tree
from decision_trees.vhdl_generators.random_forest import RandomForest
from decision_trees.utils.convert_to_fixed_point import quantize_data
from decision_trees.utils.constants import get_classifier


def test_dataset(number_of_bits_per_feature: int,
                 train_data: np.ndarray, train_target: np.ndarray,
                 test_data: np.ndarray, test_target: np.ndarray,
                 clf_type: ClassifierType,
                 ):
    # first create classifier from scikit
    clf = get_classifier(clf_type)

    # first - train the classifiers on non-quantized data
    clf.fit(train_data, train_target)
    test_predicted = clf.predict(test_data)
    print("scikit clf with test data:")
    report_performance(clf, clf_type, test_target, test_predicted)

    # perform quantization of train and test data
    # while at some point I was considering not quantizing the test data,
    # I came to a conclusion that it is not the way it will be performed in hardware
    train_data_quantized, test_data_quantized = quantize_data(train_data, test_data, number_of_bits_per_feature, True)

    clf.fit(train_data_quantized, train_target)
    test_predicted_quantized = clf.predict(test_data_quantized)
    print("scikit clf with train and test data quantized:")
    report_performance(clf, clf_type, test_target, test_predicted_quantized)

    # generate own classifier based on the one from scikit
    number_of_features = len(train_data[0])
    my_clf = generate_my_classifier(clf, number_of_features, number_of_bits_per_feature)
    my_clf_test_predicted_quantized = my_clf.predict(test_data_quantized)
    print("own clf with train and test data quantized:")
    report_performance(my_clf, clf_type, test_target, my_clf_test_predicted_quantized)

    differences_scikit_my = np.sum(test_predicted_quantized != my_clf_test_predicted_quantized)
    print(f"Number of differences between scikit_qunatized and my_quantized: {differences_scikit_my}")

    # check if own classifier works the same as scikit one
    _compare_with_own_classifier(
        [test_predicted, test_predicted_quantized, my_clf_test_predicted_quantized],
        ["scikit", "scikit_quantized", "own_clf_quantized"],
        test_target, flag_save_details_to_file=True, path="./../../data/"
    )

    # optionally check the performance of the scikit classifier for reference (does not work for own classifier)
    _test_classification_performance(clf, test_data, 10, 10)


def report_performance(clf, clf_type: ClassifierType, expected: np.ndarray, predicted: np.ndarray):
    if clf_type == ClassifierType.RANDOM_FOREST_REGRESSOR:
        _report_regressor(expected, predicted)
    else:
        _report_classifier(clf, expected, predicted)


def _report_classifier(clf, expected: np.ndarray, predicted: np.ndarray):
    print("Detailed classification report:")

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    cm = metrics.confusion_matrix(expected, predicted)
    cm = cm / cm.sum(axis=1)[:, None] * 100

    np.set_printoptions(formatter={'float': '{: 2.2f}'.format})
    print(f"Confusion matrix:\n {cm}")

    f1_score = metrics.f1_score(expected, predicted, average='weighted')
    precision = metrics.precision_score(expected, predicted, average='weighted')
    recall = metrics.recall_score(expected, predicted, average='weighted')
    accuracy = metrics.accuracy_score(expected, predicted)
    print(f"f1_score: {f1_score:{2}.{4}}")
    print(f"precision: {precision:{2}.{4}}")
    print(f"recall: {recall:{2}.{4}}")
    print(f"accuracy: {accuracy:{2}.{4}}")


def _report_regressor(expected: np.ndarray, predicted: np.ndarray):
    print("Detailed regression report:")

    mae = metrics.mean_absolute_error(expected, predicted)
    mse = metrics.mean_squared_error(expected, predicted)
    r2s = metrics.r2_score(expected, predicted)
    evs = metrics.explained_variance_score(expected, predicted)
    print(f"mean_absolute_error: {mae:{2}.{4}}")
    print(f"mean_squared_error: {mse:{2}.{4}}")
    print(f"coefficient_of_determination: {r2s:{2}.{4}}")
    print(f"explained_variance_score: {evs:{2}.{4}}")


def generate_my_classifier(clf, number_of_features, number_of_bits_per_feature: int):
    if isinstance(clf, DecisionTreeClassifier):
        print("Creating decision tree classifier!")
        my_clf = Tree("DecisionTreeClassifier", number_of_features, number_of_bits_per_feature)
    elif isinstance(clf, RandomForestClassifier):
        print("Creating random forest classifier!")
        my_clf = RandomForest("RandomForestClassifier", number_of_features, number_of_bits_per_feature)
    else:
        print("Unknown type of classifier!")
        raise ValueError("Unknown type of classifier!")

    my_clf.build(clf)
    my_clf.print_parameters()
    my_clf.create_vhdl_file("./../../data/vhdl/")

    return my_clf


def _compare_with_own_classifier(results: [], results_names: [str],
                                 test_target,
                                 flag_save_details_to_file: bool = True,
                                 path: str = "./"
                                 ):
    flag_no_errors = True
    number_of_errors = np.zeros(len(results))

    comparision_file = None
    if flag_save_details_to_file:
        comparision_file = open(path + "/comparision_details.txt", "w")

    for j in range(0, len(test_target)):
        flag_iteration_error = False

        for i in range(0, len(results)):
            if results[i][j] != test_target[j]:
                number_of_errors[i] += 1
                flag_no_errors = False
                flag_iteration_error = True

        if flag_iteration_error and flag_save_details_to_file:
            print("Difference between versions!", file=comparision_file)
            print("Ground true: " + str(test_target[j]), file=comparision_file)
            for i in range(0, len(results)):
                print(f"{results_names[i]}: {results[i][j]}", file=comparision_file)

    if flag_no_errors:
        print("All results were the same")
    else:
        for i in range(0, len(results)):
            print(f"Number of {results_names[i]} errors: {number_of_errors[i]}")


def _test_classification_performance(clf, test_data, number_of_data_to_test=1000, number_of_iterations=1000):
    if number_of_data_to_test <= len(test_data):
        start = time.clock()

        for i in range(0, number_of_iterations):
            for data in test_data[:number_of_data_to_test]:
                clf.predict([data])

        end = time.clock()
        elapsed_time = (end - start)

        print("It takes " +
              '% 2.4f' % (elapsed_time / number_of_iterations) +
              "us to classify " +
              str(number_of_data_to_test) + " data.")
    else:
        print("There is not enough data provided to evaluate the performance. It is required to provide at least " +
              str(number_of_data_to_test) + " values.")


# there is no general method for normalisation, so it was moved to be a part of each dataset
def normalise_data(train_data: np.ndarray, test_data: np.ndarray):
    from sklearn import preprocessing

    print("np.max(train_data): " + str(np.max(train_data)))
    print("np.ptp(train_data): " + str(np.ptp(train_data)))

    normalised_1 = 1 - (train_data - np.max(train_data)) / -np.ptp(train_data)
    normalised_2 = preprocessing.minmax_scale(train_data, axis=1)

    print(train_data[0])

    train_data /= 16
    test_data /= 16

    print("Are arrays equal: " + str(np.array_equal(normalised_2, train_data)))
    print("Are arrays equal: " + str(np.array_equal(normalised_1, train_data)))

    for i in range(0, 1):
        print(train_data[i])
        print(normalised_1)
        print(normalised_2)
