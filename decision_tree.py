__author__ = 'Amin'

import pickle
from analyse_classifier import get_lineage
from analyse_classifier import get_code

from sklearn.externals.six import StringIO

from Tree import Tree

import time


def visualize_tree(clf, filename):
    print "Number of splits: ", len(clf.tree_.value) #array of nodes values
    #print "Number of features: ", len(tree.tree_.feature)
    #print "Number of thresholds: ", len(tree.tree_.threshold)

    # save as *.dot file
    # with open(filename + ".dot", 'w') as f:
    #     f = tree.export_graphviz(clf, out_file=f)

    # remove file
    # import os
    # os.unlink(filename + ".dot")

    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename + ".pdf")


def visualize_forest(forest, filename_pattern):
    print "Number of trees in the forest: ", forest.n_estimators
    counter = 0
    for clf in forest.estimators_[:]:
        filename = filename_pattern + str(counter)
        visualize_tree(clf, filename)
        counter += 1


def test_classifier(clf, test_data, predicted_result):
    correct_classifications = 0.0
    incorrect_classifications = 0.0

    for histogram in test_data:
        if clf.predict(histogram) == predicted_result:
            correct_classifications += 1.0
        else:
            incorrect_classifications += 1.0
    #print "Correct classifications: ", correct_classifications
    #print "Incorrect classifications: ", incorrect_classifications

    return correct_classifications, incorrect_classifications

# prepare training data
with open("data\\positive_histograms", "rb") as f:
    train_histogram_positive = pickle.load(f)

class_labels_positive = [1] * len(train_histogram_positive)

with open("data\\test_positive_histograms", "rb") as f:
    test_histogram_positive = pickle.load(f)


with open("data\\negative_histograms", "rb") as f:
    train_histogram_negative = pickle.load(f)

class_labels_negative = [0] * len(train_histogram_negative)

with open("data\\test_negative_histograms", "rb") as f:
    test_histogram_negative = pickle.load(f)

training_data = train_histogram_positive + train_histogram_negative
class_labels = class_labels_positive + class_labels_negative

# train and test the tree
classifier = None


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#for nr_of_trees in xrange(51, 52, 10):
for nr_of_trees in xrange(2, 3):
    #for depth in xrange(1, 21):
    for depth in xrange(21, 22):
        classifier = DecisionTreeClassifier(max_depth=depth)
        classifier = classifier.fit(training_data, class_labels)

        classifier = RandomForestClassifier(n_estimators=nr_of_trees, max_depth=depth)
        classifier = classifier.fit(training_data, class_labels)

        correct_classifications = 0.0
        incorrect_classifications = 0.0

        #print "Positive tests: "
        correct_p, incorrect_p = test_classifier(classifier, test_histogram_positive, 1)
        correct_classifications += correct_p
        incorrect_classifications += incorrect_p

        #print "Negative tests: "
        correct_n, incorrect_n = test_classifier(classifier, test_histogram_negative, 0)
        correct_classifications += correct_n
        incorrect_classifications += incorrect_n

        print "Accuracy (depth: " + '% 02.0f' %(depth) + \
              ", nr of trees: " + '% 02.0f' %(nr_of_trees) + \
              "):" + \
              " overall: " + \
              '% 2.4f' %(correct_classifications / (correct_classifications + incorrect_classifications)) + \
              ", " + \
              " positive: " + \
              '% 2.4f' %(correct_p / (correct_p + incorrect_p)) + \
              ", " + \
              " negative: " + \
              '% 2.4f' %(correct_n / (correct_n + incorrect_n)) + \
              ""

# time measurements
start = time.clock()

for i in xrange(0, 10):
    for histogram in test_histogram_positive[:10]:
        classifier.predict(histogram)

end = time.clock()
time = (end - start)

print str(time) + " per classification in us"

list_of_input_value_names = []
# TODO - this value should be taken automatically
#for i in xrange(0, 3540):
for i in xrange(0, 6000):
    list_of_input_value_names.append(i)

#get_lineage(classifier.estimators_[0], list_of_input_value_names)
#get_code(classifier.estimators_[0], list_of_input_value_names)

#get_lineage(classifier, list_of_input_value_names)
#get_code(classifier, list_of_input_value_names)

if isinstance(classifier, DecisionTreeClassifier):
    print "Decision tree classifier!"
elif isinstance(classifier, RandomForestClassifier):
    print "Random forest classifier!"

    random_forest = []

    for tree in classifier.estimators_:
        tree_builder = Tree()
        tree_builder.build(tree, list_of_input_value_names)

        #tree_builder.print_leaves()
        #tree_builder.print_splits()
        print "Depth: ", tree_builder.find_depth()
        print "Number of splits: ", len(tree_builder.splits)
        print "Number of leaves: ", len(tree_builder.leaves)

        #tree_builder.create_vhdl_code("sample_dataset.vhdl")

        random_forest.append(tree)

    for histogram in test_histogram_positive:
        scikit_learn_result = classifier.predict(histogram)

        results = [0, 0]
        for tree in random_forest:
            tree_result = tree.predict(histogram)
            results[int(tree_result)] += 1

        if results[0] >= results[1]:
            my_result = 0
        else:
            my_result = 1

        if scikit_learn_result != my_result:
            print "Error!"

else:
    print "Unknown type of classifier!"





#from inspect import getmembers
#print( getmembers( classifier.estimators_[0].tree_ ) )

#visualize_forest(classifier, "tree_visualization\\tree")
#visualize_tree(classifier, "tree_visualization\\tree")

