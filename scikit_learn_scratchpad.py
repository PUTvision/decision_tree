__author__ = 'Amin'

# classification
# training data
training_data = [[0, 0], [1, 1]]
class_labels = [0, 1]

# decision tree
# http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree

tree_classifier = tree.DecisionTreeClassifier()
tree_classifier = tree_classifier.fit(training_data, class_labels)
print tree_classifier.predict([0.51, 0.51])

# forest of randomized trees
# http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=10)
forest_classifier = forest_classifier.fit(training_data, class_labels)
print forest_classifier.predict([0.51, 0.51])
