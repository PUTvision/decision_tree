import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics


def report_classifier(clf, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


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


def test_dataset(number_of_classes: int, train_data: np.ndarray, train_target: np.ndarray, test_data: np.ndarray, test_target: np.ndarray):
    clf_decision_tree = DecisionTreeClassifier()  # max_depth=50)
    clf_decision_tree.fit(train_data, train_target)
    test_predicted = clf_decision_tree.predict(test_data)
    report_classifier(clf_decision_tree, test_target, test_predicted)

    from tree import Tree
    my_clf_decision_tree = Tree("TreeTest", number_of_classes)
    my_clf_decision_tree.build(clf_decision_tree)
    my_clf_decision_tree.print_parameters()
    my_clf_decision_tree.create_vhdl_file()

    compare_with_own_classifier(clf_decision_tree, my_clf_decision_tree, test_data)

    clf_random_forest = RandomForestClassifier()  # n_estimators=10
    clf_random_forest.fit(train_data, train_target)
    test_predicted = clf_random_forest.predict(test_data)
    report_classifier(clf_decision_tree, test_target, test_predicted)

    from tree import RandomForest
    my_clf_random_forest = RandomForest(number_of_classes)
    my_clf_random_forest.build(clf_random_forest)
    my_clf_random_forest.print_parameters()
    my_clf_random_forest.create_vhdl_file()

    compare_with_own_classifier(clf_random_forest, my_clf_random_forest, test_data)
