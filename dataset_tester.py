import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from tree import Tree
from tree import RandomForest

import time


def report_classifier(clf, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    correct_classifications = 0
    incorrect_classifications = 0
    for e, p in zip(expected, predicted):
        if e == p:
            correct_classifications += 1
        else:
            incorrect_classifications += 1
    print("Accuracy overall: " +
          '% 2.4f' % (correct_classifications / (correct_classifications + incorrect_classifications))
          )


def compare_with_own_classifier(scikit_clf, own_clf, test_data):
    flag_no_errors = True
    number_of_errors = 0
    for sample in test_data:
        scikit_result = scikit_clf.predict([sample])
        my_result = own_clf.predict(sample)

        if scikit_result != my_result:
            print("Error!")
            print(scikit_result)
            print(my_result)
            number_of_errors += 1
            flag_no_errors = False

    if flag_no_errors:
        print("All results were the same")
    else:
        print("Number of errors: " + str(number_of_errors))


def test_dataset(number_of_bits_per_feature: int,
                 train_data: np.ndarray, train_target: np.ndarray,
                 test_data: np.ndarray, test_target: np.ndarray
                 ):
    number_of_features = len(train_data[0])

    clf_decision_tree = DecisionTreeClassifier()  # max_depth=50)
    clf_decision_tree.fit(train_data, train_target)
    test_predicted = clf_decision_tree.predict(test_data)
    report_classifier(clf_decision_tree, test_target, test_predicted)

    from tree import Tree
    my_clf_decision_tree = Tree("TreeTest", number_of_features, number_of_bits_per_feature)
    my_clf_decision_tree.build(clf_decision_tree)
    my_clf_decision_tree.print_parameters()
    my_clf_decision_tree.create_vhdl_file()

    compare_with_own_classifier(clf_decision_tree, my_clf_decision_tree, test_data)

    clf_random_forest = RandomForestClassifier()  # n_estimators=10
    clf_random_forest.fit(train_data, train_target)
    test_predicted = clf_random_forest.predict(test_data)
    report_classifier(clf_decision_tree, test_target, test_predicted)

    from tree import RandomForest
    my_clf_random_forest = RandomForest(number_of_features, number_of_bits_per_feature)
    my_clf_random_forest.build(clf_random_forest)
    my_clf_random_forest.print_parameters()
    my_clf_random_forest.create_vhdl_file()

    compare_with_own_classifier(clf_random_forest, my_clf_random_forest, test_data)


def test_classification_performance(clf, test_data, number_of_data_to_test=1000, number_of_iterations=1000):
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


def normalise_data(train_data: np.ndarray, test_data: np.ndarray):
    from sklearn import preprocessing

    print("np.max(train_data): " + str(np.max(train_data)))
    print("np.ptp(train_data): " + str(np.ptp(train_data)))

    normalised_1 = 1 - (train_data - np.max(train_data)) / -np.ptp(train_data)
    normalised_2 = preprocessing.minmax_scale(train_data)

    train_data /= 16
    test_data /= 16

    print("Are arrays equal: " + str(np.array_equal(normalised_2, train_data)))
    print("Are arrays equal: " + str(np.array_equal(normalised_1, train_data)))

    for i in range(0, 1):
        print(train_data[i])
        print(normalised_1)
        print(normalised_2)


def generate_my_classifier(classifier, test_data):
    number_of_features = len(test_data[0])

    if isinstance(classifier, DecisionTreeClassifier):
        print("Decision tree classifier!")
        my_classifier = Tree("HoG_tree", number_of_features, 16)

    elif isinstance(classifier, RandomForestClassifier):
        print("Random forest classifier!")
        my_classifier = RandomForest("HoG_forest", number_of_features, 8)

    else:
        print("Unknown type of classifier!")

    my_classifier.build(classifier)
    my_classifier.print_parameters()

    compare_with_own_classifier(classifier, my_classifier, test_data)


def grid_search(train_data, train_target, test_data, test_target):
    # perform grid search to find best parameters
    # TODO - think about which metric would be best
    scores = ['f1']#''precision', 'recall']

    tuned_parameters = [{'max_depth': [5, 10]}]

    clf = None

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='%s_macro' % score)
        clf = clf.fit(train_data, train_target)

        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        predicted_class_labels = clf.predict(test_data)
        report_classifier(clf, test_target, predicted_class_labels)

        print()
