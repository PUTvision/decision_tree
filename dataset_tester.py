import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

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


def test_dataset(number_of_features: int, number_of_bits_per_feature: int,
                 train_data: np.ndarray, train_target: np.ndarray,
                 test_data: np.ndarray, test_target: np.ndarray
                 ):
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
