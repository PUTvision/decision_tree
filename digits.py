from sklearn import datasets
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import numpy as np


def simple_test():
    # Load the digits dataset
    digits = datasets.load_digits()
    print(digits.data.shape)

    # Display the first digit
    plt.figure(1, figsize=(5, 5))
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


def sample_from_scikit(classifier):
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

    #plt.show()


def mnist_processing(clf, dataset):
    train_samples = 60000
    test_samples = 10000

    data_target = list(zip(dataset.data[:60000], dataset.target[:60000]))
    import random
    random.shuffle(data_target)

    d, t = zip(*data_target)

    train_data = np.array(d[:train_samples]).reshape((train_samples, -1))
    print(len(train_data))
    train_target = np.array(t[:train_samples])
    print(len(train_target))

    test_data = dataset.data[-test_samples:].reshape((test_samples, -1))
    print(len(test_data))
    test_target = dataset.target[-test_samples:]
    print(len(test_target))

    clf.fit(train_data, train_target)

    predicted = clf.predict(test_data)
    expected = test_target

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


def load_and_compare_datasets():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home=".//data//MNIST//")
    print(mnist.data.shape)
    print(mnist.target.shape)
    print(np.unique(mnist.target))

    digits = datasets.load_digits()
    print(digits.data.shape)
    print(digits.target.shape)
    print(np.unique(digits.target))
    print(len(digits.images))

    return mnist, digits


if __name__ == "__main__":

    #clf = DecisionTreeClassifier()#max_depth=50)
    #clf = svm.SVC(gamma=0.001)
    clf = RandomForestClassifier(n_estimators=10)

    sample_from_scikit(clf)

    mnist, digits = load_and_compare_datasets()
    #mnist_processing(clf, mnist)

    list_of_input_value_names = []
    for i in range(0, 64):
        list_of_input_value_names.append(i)

    #from tree import Tree
    #my_classifier = Tree("TreeTest", len(list_of_input_value_names))

    from tree import RandomForest
    my_classifier = RandomForest(len(list_of_input_value_names))

    my_classifier.build(clf, list_of_input_value_names)

    my_classifier.print_parameters()

    # from analyse_classifier import tree_to_code
    # feature_names = []
    # for i in range(64):
    #     feature_names.append(str(i))
    # tree_to_code(clf, feature_names)

    n_samples = len(digits.images)
    data = digits.data.reshape((n_samples, -1))

    for digit in data[n_samples // 2:n_samples // 2+10]:
        scikit_learn_result = clf.predict([digit])
        my_result = my_classifier.predict(digit)

        if scikit_learn_result != my_result:
            print("Error!")
            print(scikit_learn_result)
            print(my_result)

    my_classifier.create_vhdl_file()
