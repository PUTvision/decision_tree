import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import dataset_tester


def load_data(data_filename, nr_pos_train: int, nr_pos_test: int, nr_neg_train: int, nr_neg_test: int):
    # prepare the training data
    with open("..\\data\\positive_train_" + str(data_filename) + ".pickle", "rb") as f:
        train_data_positive = pickle.load(f)

    with open("..\\data\\positive_test_" + str(data_filename) + ".pickle", "rb") as f:
        test_data_positive = pickle.load(f)

    with open("..\\data\\negative_train_" + str(data_filename) + ".pickle", "rb") as f:
        train_data_negative = pickle.load(f)

    with open("..\\data\\negative_test_" + str(data_filename) + ".pickle", "rb") as f:
        test_data_negative = pickle.load(f)

    train_data = train_data_positive[0:nr_pos_train] + train_data_negative[0:nr_neg_train]
    train_target = [1] * nr_pos_train + [0] * nr_neg_train

    test_data = test_data_positive[0:nr_pos_test] + test_data_negative[0:nr_neg_test]
    test_target = [1] * nr_pos_test + [0] * nr_neg_test

    return train_data, train_target, test_data, test_target


def test_inria_hog():
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

    dataset_tester.test_dataset(4,
                                train_data, train_target, test_data, test_target,
                                dataset_tester.ClassifierType.decision_tree
                                )

    assert True
