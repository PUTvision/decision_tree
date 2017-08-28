import numpy as np

from sklearn import datasets
from sklearn import svm, metrics

import matplotlib.pyplot as plt

import dataset_tester


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


def load_data(number_of_train_samples: int,
              number_of_test_samples: int,
              ):
    digits = datasets.load_digits()
    # print(digits.data.shape)
    # print(digits.target.shape)
    # print(np.unique(digits.target))

    # data has to be flatten (8x8 image -> 64x1 matrix)
    data = digits.data.reshape((len(digits.data), -1))
    # print(len(data))

    train_data = data[:number_of_train_samples]
    train_target = digits.target[:number_of_train_samples]
    test_data = data[number_of_train_samples:number_of_train_samples+number_of_test_samples]
    test_target = digits.target[number_of_train_samples:number_of_train_samples+number_of_test_samples]

    return train_data, train_target, test_data, test_target

if __name__ == "__main__":
    # sample_from_scikit()

    #####################################
    # SET THE FOLLOWING PARAMETERS
    # DIGITS DATABASE
    # total number of samples: 1797 (each is 8x8)
    number_of_train_samples = 1200
    number_of_test_samples = 1797 - number_of_train_samples
    # END OF PARAMETERS SETTING
    # sanity check
    if (number_of_train_samples + number_of_test_samples) > 1797:
        print("ERROR, too much samples set!")
    #####################################

    train_data, train_target, test_data, test_target = load_data(
        number_of_train_samples,
        number_of_test_samples
    )

    # TODO - it is neccessary to add normalisation step here. otherwise the input is not in 0-1 range
    # TODO cont. - thus not taking into account bit per feature (which works only for fractions
    # TODO dataset_tester.normalise_data

    # TODO - add option to change the input data to some number of bits so that is can also be compared with full resolution

    dataset_tester.test_dataset(4, train_data, train_target, test_data, test_target)
