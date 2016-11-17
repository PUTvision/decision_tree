__author__ = 'Amin'

# http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

import numpy as np


def get_code(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node):
        if (threshold[node] != -2):
            print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
            if left[node] != -1:
                recurse (left, right, threshold, features,left[node])
            print "} else {"
            if right[node] != -1:
                recurse(left, right, threshold, features,right[node])
            print "}"
        else:
            print "return " + str(value[node])

    recurse(left, right, threshold, features, 0)


def get_lineage(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    for child in idx:
        for node in recurse(left, right, child):
            print node

if __name__ == "__main__":
    # sample usage for random forest classifier to show first tree
    #get_lineage(classifier.estimators_[0], list_of_input_value_names)
    #get_code(classifier.estimators_[0], list_of_input_value_names)

    #get_lineage(classifier, list_of_input_value_names)
    #get_code(classifier, list_of_input_value_names)
    pass