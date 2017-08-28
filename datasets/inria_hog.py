import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import dataset_tester


def load_data(data_filename, nr_pos_train, nr_pos_test, nr_neg_train, nr_neg_test):
    # prepare the training data
    with open("data\\positive_train_" + str(data_filename) + ".pickle", "rb") as f:
        train_data_positive = pickle.load(f)

    with open("data\\positive_test_" + str(data_filename) + ".pickle", "rb") as f:
        test_data_positive = pickle.load(f)

    with open("data\\negative_train_" + str(data_filename) + ".pickle", "rb") as f:
        train_data_negative = pickle.load(f)

    with open("data\\negative_test_" + str(data_filename) + ".pickle", "rb") as f:
        test_data_negative = pickle.load(f)

    train_data = train_data_positive[0:nr_pos_train] + train_data_negative[0:nr_neg_train]
    train_target = [1] * nr_pos_train + [0] * nr_neg_train

    test_data = test_data_positive[0:nr_pos_test] + test_data_negative[0:nr_neg_test]
    test_target = [1] * nr_pos_test + [0] * nr_neg_test

    return train_data, train_target, test_data, test_target


if __name__ == "__main__":
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # INRIA DATABASE FOR HOG (64x128)
    # total number of positive samples: 1126, but only 1100 can be used here (900 for samples, 200 for tests)
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
    # train_data, train_target, test_data, test_target = load_data("samples_modified",
    #                                                              number_of_positive_samples,
    #                                                              number_of_positive_tests,
    #                                                              number_of_negative_samples,
    #                                                              number_of_negative_tests
    #                                                              )
    print("Result with HoG from skimage:")
    train_data, train_target, test_data, test_target = load_data("samples",
                                                                 number_of_positive_samples,
                                                                 number_of_positive_tests,
                                                                 number_of_negative_samples,
                                                                 number_of_negative_tests
                                                                 )

    clf = DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)

    # clf = RandomForestClassifier(n_estimators=nr_of_trees, max_depth=depth)
    # clf = classifier.fit(train_data, train_target)
    #
    # clf = LinearSVC()
    # clf = classifier.fit(train_data, train_target)

    dataset_tester.generate_my_classifier(clf, test_data)

    # Use this to test the performance (speed of execution)
    dataset_tester.test_classification_performance(clf, test_data, number_of_data_to_test=100)
