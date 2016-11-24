__author__ = 'Amin'

import pickle
from analyse_classifier import get_lineage
from analyse_classifier import get_code

from sklearn.externals.six import StringIO

from Tree import Tree
from Tree import RandomForest


def visualize_tree(clf, filename):
    print("Number of splits: ", len(clf.tree_.value)) #array of nodes values
    print("Number of features: ", len(clf.tree.tree_.feature))
    print("Number of thresholds: ", len(clf.tree.tree_.threshold))

    # save as *.dot file
    # with open(filename + ".dot", 'w') as f:
    #     f = tree.export_graphviz(clf, out_file=f)

    # remove file
    # import os
    # os.unlink(filename + ".dot")

    import pydot
    dot_data = StringIO()
    clf.tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename + ".pdf")


def visualize_forest(forest, filename_pattern):
    print("Number of trees in the forest: ", forest.n_estimators)
    counter = 0
    for clf in forest.estimators_[:]:
        filename = filename_pattern + str(counter)
        visualize_tree(clf, filename)
        counter += 1


def check_and_print_classifier_accuracy(clf, test_data_positive, test_data_negative):
    correct_classifications = 0.0
    incorrect_classifications = 0.0

    #print "Positive tests: "
    correct_p, incorrect_p = test_classifier(clf, test_data_positive, 1)
    correct_classifications += correct_p
    incorrect_classifications += incorrect_p

    #print "Negative tests: "
    correct_n, incorrect_n = test_classifier(clf, test_data_negative, 0)
    correct_classifications += correct_n
    incorrect_classifications += incorrect_n

    print("Accuracy: " +
          " overall: " +
          '% 2.4f' % (correct_classifications / (correct_classifications + incorrect_classifications)) + \
          ", " +
          " positive: " +
          '% 2.4f' % (correct_p / (correct_p + incorrect_p)) +
          ", " +
          " negative: " +
          '% 2.4f' % (correct_n / (correct_n + incorrect_n)) +
          "")


def test_classifier(clf, test_data, predicted_result):
    correct_classifications = 0.0
    incorrect_classifications = 0.0

    for data in test_data:
        # [something] is required as otherwise it is treated as 1d array, which generates warnings in scikit learn
        if clf.predict([data]) == predicted_result:
            correct_classifications += 1.0
        else:
            incorrect_classifications += 1.0
    #print "Correct classifications: ", correct_classifications
    #print "Incorrect classifications: ", incorrect_classifications

    return correct_classifications, incorrect_classifications


def test_classification_performance(clf, test_data):
    import time

    number_of_iterations = 1000
    number_of_data_to_test = 1000

    if number_of_data_to_test < len(test_data):
        start = time.clock()

        for i in range(0, number_of_iterations):
            for data in test_data[:number_of_data_to_test]:
                clf.predict(data)

        end = time.clock()
        elapsed_time = (end - start)

        print("It takes " + str(elapsed_time / number_of_iterations) + "us to classify " + \
              str(number_of_data_to_test) + " data.")
    else:
        print("There is not enough data provided to evaluate the performance. It is required to provide at least " + \
              str(number_of_data_to_test) + "values.")


def generate_my_classifier(classifier):
    list_of_input_value_names = []
    # TODO - this value should be taken automatically
    #for i in xrange(0, 3540):
    for i in range(0, 6000):
        list_of_input_value_names.append(i)

    if isinstance(classifier, DecisionTreeClassifier):
        print("Decision tree classifier!")
        my_classifier = Tree()

    elif isinstance(classifier, RandomForestClassifier):
        print("Random forest classifier!")
        my_classifier = RandomForest()

    else:
        print("Unknown type of classifier!")

    my_classifier.build(classifier, list_of_input_value_names)

    my_classifier.print_parameters()

    for histogram in test_data_positive:
        scikit_learn_result = classifier.predict(histogram)
        my_result = my_classifier.predict(histogram)

        if scikit_learn_result != my_result:
            print("Error!")

    for histogram in test_data_negative:
        scikit_learn_result = classifier.predict(histogram)
        my_result = my_classifier.predict(histogram)

        if scikit_learn_result != my_result:
            print("Error!")

    #my_classifier.create_vhdl_code_old("tree.vhdl")
    my_classifier.create_vhdl_file()

    #from inspect import getmembers
    #print( getmembers( classifier.estimators_[0].tree_ ) )

    #visualize_forest(classifier, "tree_visualization\\tree")
    #visualize_tree(classifier, "tree_visualization\\tree")

if __name__ == "__main__":

    # version for HOG
    data_filename = "data"
    # version for LBP
    #data_filename = "histograms"

    # prepare the training data
    with open("data\\positive_" + str(data_filename), "rb") as f:
        train_data_positive = pickle.load(f)

    class_labels_positive = [1] * len(train_data_positive)

    with open("data\\test_positive_" + str(data_filename), "rb") as f:
        test_data_positive = pickle.load(f)

    with open("data\\negative_" + str(data_filename), "rb") as f:
        train_data_negative = pickle.load(f)

    class_labels_negative = [0] * len(train_data_negative)

    with open("data\\test_negative_" + str(data_filename), "rb") as f:
        test_data_negative = pickle.load(f)

    training_data = train_data_positive + train_data_negative
    class_labels = class_labels_positive + class_labels_negative

    # train and test the classifier
    classifier = None

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC

    #for nr_of_trees in range(51, 52, 10):
    for nr_of_trees in range(2, 3):
        #for depth in range(1, 21):
        for depth in range(21, 22):

            print("Parameters: depth: " + '% 02.0f' % depth +
                  ", nr of trees: " + '% 02.0f' % nr_of_trees +
                  "): ")

            # classifier = DecisionTreeClassifier(max_depth=depth)
            # classifier = classifier.fit(training_data, class_labels)

            #classifier = RandomForestClassifier(n_estimators=nr_of_trees, max_depth=depth)
            #classifier = classifier.fit(training_data, class_labels)

            classfier = LinearSVC()
            classifier = classfier.fit(training_data, class_labels)

            check_and_print_classifier_accuracy(classifier, test_data_positive, test_data_negative)

    # Use this to test the performance (speed of execution)
    #test_classification_performance(classifier, test_data_positive)

    #generate_my_classifier(classfier)
