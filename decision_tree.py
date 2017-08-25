import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV

from tree import Tree
from tree import RandomForest

import dataset_tester


def train_classifier_and_test(data_filename, nr_pos_train, nr_pos_test, nr_neg_train, nr_neg_test):
    # prepare the training data
    with open("data\\positive_train_" + str(data_filename) + ".pickle", "rb") as f:
        train_data_positive = pickle.load(f)

    with open("data\\positive_test_" + str(data_filename) + ".pickle", "rb") as f:
        test_data_positive = pickle.load(f)

    with open("data\\negative_train_" + str(data_filename) + ".pickle", "rb") as f:
        train_data_negative = pickle.load(f)

    with open("data\\negative_test_" + str(data_filename) + ".pickle", "rb") as f:
        test_data_negative = pickle.load(f)

    training_data = train_data_positive[0:nr_pos_train] + train_data_negative[0:nr_neg_train]
    training_class_labels = [1] * nr_pos_train + [0] * nr_neg_train

    test_data = test_data_positive[0:nr_pos_test] + test_data_negative[0:nr_neg_test]
    test_class_labels = [1] * nr_pos_test + [0] * nr_neg_test

    # perform grid search to find best parameters
    # TODO - think about which metric would be best
    # scores = ['f1']#''precision', 'recall']
    #
    # tuned_parameters = [{'max_depth': [5, 10]}]
    #
    # clf = None

    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    #     clf = clf.fit(training_data, training_class_labels)
    #
    #     print("Best parameters set found on development set:")
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    #     print()
    #
    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     predicted_class_labels = clf.predict(test_data)
    #     dataset_tester.report_classifier(clf, test_class_labels, predicted_class_labels)
    #
    #     print()

        #classifier = RandomForestClassifier(n_estimators=nr_of_trees, max_depth=depth)
        #classifier = classifier.fit(training_data, class_labels)

        #classifier = LinearSVC()
        #classifier = classifier.fit(training_data, class_labels)

    # Use this to test the performance (speed of execution)
    #dataset_tester.test_classification_performance(classifier, test_data, number_of_data_to_test=100)

    clf = DecisionTreeClassifier()
    clf = clf.fit(training_data, training_class_labels)

    generate_my_classifier(clf, test_data)


def generate_my_classifier(classifier, test_data):

    if isinstance(classifier, DecisionTreeClassifier):
        print("Decision tree classifier!")
        # number of feature for HoG should be 3780
        my_classifier = Tree("HoG_tree", len(test_data[0]), 16)

    elif isinstance(classifier, RandomForestClassifier):
        print("Random forest classifier!")
        my_classifier = RandomForest("HoG_forest", 3780, 8)

    else:
        print("Unknown type of classifier!")

    my_classifier.build(classifier)
    my_classifier.print_parameters()

    dataset_tester.compare_with_own_classifier(classifier, my_classifier, test_data)


if __name__ == "__main__":
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # INRIA DATABASE FOR HOG (64x128)
    # total number of positive samples: 1126, but only 1100 can be used here
    number_of_positive_samples = 200#900
    number_of_positive_tests = 50#200
    # total number of negative train samples: 1218, but only 1200 can be used here
    number_of_negative_samples = 400#1200
    # total number of negative test samples: 453, , but only 400 can be used here
    number_of_negative_tests = 50#400
    # END OF PARAMETERS SETTING
    #####################################

    # version for HOG
    # print("Result with HoG modified:")
    # train_classifier_and_test("samples_modified",
    #                           number_of_positive_samples,
    #                           number_of_positive_tests,
    #                           number_of_negative_samples,
    #                           number_of_negative_tests
    #                           )
    print("Result with HoG from skimage:")
    train_classifier_and_test("samples",
                              number_of_positive_samples,
                              number_of_positive_tests,
                              number_of_negative_samples,
                              number_of_negative_tests
                              )
