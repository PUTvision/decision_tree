from sklearn import datasets
from sklearn.utils import shuffle

import dataset_tester


def load_data(number_of_train_samples: int, number_of_test_samples: int):
    mnist = datasets.fetch_mldata('MNIST original', data_home=".//data//MNIST//")
    # print(mnist.data.shape)
    # print(mnist.target.shape)
    # print(np.unique(mnist.target))

    # TODO - is this enough or should I use my own function?
    mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
    # shuffle_data_and_target(mnist.data, mnist.target)

    train_data = mnist.data[:number_of_train_samples]
    train_target = mnist.target[:number_of_train_samples]
    test_data = mnist.data[number_of_train_samples:number_of_train_samples+number_of_test_samples]
    test_target = mnist.target[number_of_train_samples:number_of_train_samples+number_of_test_samples]

    return train_data, train_target, test_data, test_target


def test_mnist_raw():
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # MNIST DATABASE
    # total number of samples: 70000 (each is 28x28)
    number_of_train_samples = 1000  # 65000
    number_of_test_samples = 100  # 70000 - number_of_train_samples
    # END OF PARAMETERS SETTING
    if (number_of_train_samples + number_of_test_samples) > 70000:
        print("ERROR, too much samples set!")
    #####################################

    train_data, train_target, test_data, test_target = load_data(
        number_of_train_samples,
        number_of_test_samples
    )

    dataset_tester.test_dataset(4,
                                train_data, train_target, test_data, test_target,
                                dataset_tester.ClassifierType.decison_tree
                                )

    assert True
