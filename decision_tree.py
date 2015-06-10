__author__ = 'Amin'

import pickle


def visualize_forest(forest):
    print "Number of trees in the forest: ", forest.n_estimators
    counter = 0
    for clf in forest.estimators_[:]:
        print "Number of splits: ", len(clf.tree_.value) #array of nodes values
        #print "Number of features: ", len(clf.tree_.feature)
        #print "Number of thresholds: ", len(clf.tree_.threshold)
        filename = "tree_visualization\\iris" + str(counter)

        from sklearn.externals.six import StringIO

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

        counter += 1


def test_classifier(clf, test_data, predicted_result):
    correct_classifications = 0
    incorrect_classifications = 0

    for histogram in test_data:
        if clf.predict(histogram) == predicted_result:
            correct_classifications += 1
        else:
            incorrect_classifications += 1
    print "Correct classifications: ", correct_classifications
    print "Incorrect classifications: ", incorrect_classifications

# prepare training data
with open("positive_histograms", "rb") as f:
    train_histogram_positive = pickle.load(f)

class_labels_positive = [1] * len(train_histogram_positive)

with open("test_positive_histograms", "rb") as f:
    test_histogram_positive = pickle.load(f)


with open("negative_histograms", "rb") as f:
    train_histogram_negative = pickle.load(f)

class_labels_negative = [0] * len(train_histogram_negative)

with open("test_negative_histograms", "rb") as f:
    test_histogram_negative = pickle.load(f)

training_data = train_histogram_positive + train_histogram_negative
class_labels = class_labels_positive + class_labels_negative

# train and test the tree
from sklearn import tree
# classifier = tree.DecisionTreeClassifier()
# classifier = classifier.fit(training_data, class_labels)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)
classifier = classifier.fit(training_data, class_labels)

# print "Positive train: "
# test_classifier(classifier, train_histogram_positive, 1)

print "Positive tests: "
test_classifier(classifier, test_histogram_positive, 1)

# print "Negative train: "
# test_classifier(classifier, train_histogram_negative, 0)

print "Negative tests: "
test_classifier(classifier, test_histogram_negative, 0)

#from inspect import getmembers
#print( getmembers( classifier.tree_ ) )

#visualize_forest(classifier)

