import unittest

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import glob
import os


def visualize_tree(clf: DecisionTreeClassifier, filename: str):
    print("Number of splits: ", len(clf.tree_.value))  # array of nodes values
    print("Number of features: ", len(clf.tree_.feature))
    print("Number of thresholds: ", len(clf.tree_.threshold))

    # save as *.dot file
    with open(filename + ".dot", 'w') as f:
        sklearn.tree.export_graphviz(clf, out_file=f)

    # TODO this part does not work atm, as I do not have pydot installed
    # import pydot
    # dot_data = StringIO()
    # clf.tree.export_graphviz(clf, out_file=dot_data)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf(filename + ".pdf")


def visualize_forest(random_forest: RandomForestClassifier, filename_pattern: str):
    print("Number of trees in the forest: ", random_forest.n_estimators)
    counter = 0
    for clf in random_forest.estimators_[:]:
        filename = filename_pattern + str(counter)
        visualize_tree(clf, filename)
        counter += 1


class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.training_data = [[0, 0], [1, 1]]
        self.class_labels = [0, 1]

    def test_tree_visualization(self):
        tree_classifier = DecisionTreeClassifier()
        tree_classifier = tree_classifier.fit(self.training_data, self.class_labels)
        visualize_tree(tree_classifier, "tree_test")

        flag_file_exist = os.path.isfile("tree_test.dot")
        self.assertTrue(flag_file_exist)

    def test_random_forest_visualization(self):
        n_estimators = 10

        forest_classifier = RandomForestClassifier(n_estimators=n_estimators)
        forest_classifier = forest_classifier.fit(self.training_data, self.class_labels)
        visualize_forest(forest_classifier, "forest_test")

        for i in range(0, n_estimators):
            flag_file_exist = os.path.isfile("forest_test" + str(i) + ".dot")
            self.assertTrue(flag_file_exist)

    def tearDown(self):
        # remove files
        file_list = glob.glob("*.dot")
        for f in file_list:
            os.remove(f)

if __name__ == "__main__":
    unittest.main()
