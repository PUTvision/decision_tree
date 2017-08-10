
import numpy as np

from sklearn import datasets
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


def sample_from_scikit():
    # The digits dataset
    digits = datasets.load_digits()

    # The data that we are interested in is made of 8x8 images of digits, let's
    # have a look at the first 4 images, stored in the `images` attribute of the
    # dataset.  If we were working from image files, we could load them using
    # matplotlib.pyplot.imread.  Note that each image must have the same size. For these
    # images, we know which digit they represent: it is given in the 'target' of
    # the dataset.

    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.data.reshape((n_samples, -1))

    # We learn the digits on the first half of the digits
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)

    plt.show()


def report_classifier(clf, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


def compare_with_own_classifier(scikit_clf, own_clf, test_data):
    flag_no_errors = True
    number_of_errors = 0
    for sample in test_data:
        scikit_result = scikit_clf.predict([sample])
        my_result = own_clf.predict(sample)

        if scikit_result != my_result:
            print("Error!")
            print(scikit_result)
            print(my_result)
            number_of_errors += 1
            flag_no_errors = False

    if flag_no_errors:
        print("All results were the same")
    else:
        print("Number of errors: " + str(number_of_errors))

if __name__ == "__main__":
    # sample_from_scikit()

    digits = datasets.load_digits()
    print(digits.data.shape)
    print(digits.target.shape)
    print(np.unique(digits.target))

    # data has to be flatten (8x8 image -> 64x1 matrix)
    data = digits.data.reshape((len(digits.data), -1))
    print(len(data))

    #####################################
    # SET THE FOLLOWING PARAMETERS
    # DIGITS DATABASE
    # total number of samples: 1797
    number_of_train_samples = 1200
    number_of_test_samples = 597
    # END OF PARAMETERS SETTING
    #####################################

    # sanity check
    if (number_of_train_samples + number_of_test_samples) > len(digits.data):
        print("ERROR, too much samples set!")

    train_data = data[:number_of_train_samples]
    train_target = digits.target[:number_of_train_samples]
    test_data = data[number_of_train_samples:]
    test_target = digits.target[number_of_train_samples:]

    clf_decision_tree = DecisionTreeClassifier()#max_depth=50)
    clf_decision_tree.fit(train_data, train_target)
    test_predicted = clf_decision_tree.predict(test_data)
    report_classifier(clf_decision_tree, test_target, test_predicted)

    from tree import Tree
    my_clf_decision_tree = Tree("TreeTest", digits.data.shape[1])
    my_clf_decision_tree.build(clf_decision_tree)
    my_clf_decision_tree.print_parameters()
    my_clf_decision_tree.create_vhdl_file()

    compare_with_own_classifier(clf_decision_tree, my_clf_decision_tree, test_data)

    clf_random_forest = RandomForestClassifier()#n_estimators=10
    clf_random_forest.fit(train_data, train_target)
    test_predicted = clf_random_forest.predict(test_data)
    report_classifier(clf_decision_tree, test_target, test_predicted)

    from tree import RandomForest
    my_clf_random_forest = RandomForest(digits.data.shape[1])
    my_clf_random_forest.build(clf_random_forest)
    my_clf_random_forest.print_parameters()
    my_clf_random_forest.create_vhdl_file()

    # TODO - there are errors in classification!!! Check it
    compare_with_own_classifier(clf_random_forest, my_clf_random_forest, test_data)
