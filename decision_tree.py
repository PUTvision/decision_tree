__author__ = 'Amin'

import pickle

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
# for histogram in train_histogram_positive:
#     print classifier.predict(histogram)

correct_classifications = 0
incorrect_classifications = 0

print "Positive tests: "
for histogram in test_histogram_positive:
    if classifier.predict(histogram) == 1:
        correct_classifications += 1
    else:
        incorrect_classifications += 1
print "Correct classifications: ", correct_classifications
print "Incorrect classifications: ", incorrect_classifications

correct_classifications = 0
incorrect_classifications = 0

# print "Negative train: "
# for histogram in train_histogram_negative:
#     print classifier.predict(histogram)

print "Negative tests: "
for histogram in test_histogram_negative:
    if classifier.predict(histogram) == 0:
        correct_classifications += 1
    else:
        incorrect_classifications += 1
print "Correct classifications: ", correct_classifications
print "Incorrect classifications: ", incorrect_classifications

#from inspect import getmembers
#print( getmembers( classifier.tree_ ) )

print "Number of trees in the forest: ", classifier.n_estimators
counter = 0
for clf in classifier.estimators_[:]:
    print "Number of splits: ", len(clf.tree_.value) #array of nodes values
    #print "Number of features: ", len(clf.tree_.feature)
    #print "Number of thresholds: ", len(clf.tree_.threshold)
    filename = "iris" + str(counter)

    from sklearn.externals.six import StringIO
    with open(filename + ".dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename + ".pdf")
    import os
    os.unlink(filename + ".dot")

    counter += 1