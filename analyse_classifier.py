from sklearn.tree import _tree

# http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if feature[{}] <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


if __name__ == "__main__":
    import sklearn.datasets

    digits = sklearn.datasets.load_digits()
    data = digits.data.reshape((len(digits.images), -1))

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(data, digits.target)

    feature_names = []
    for i in range(64):
        feature_names.append(str(i))

    tree_to_code(clf, feature_names)

    # for random forest classifier use the abov function for each tree
    # sample for first tree:
    # tree_to_code(clf.estimators_[0], feature_names)
