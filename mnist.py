import numpy as np

from sklearn import datasets

from dataset_tester import test_dataset


if __name__ == "__main__":

    mnist = datasets.fetch_mldata('MNIST original', data_home=".//data//MNIST//")
    print(mnist.data.shape)
    print(mnist.target.shape)
    print(np.unique(mnist.target))

    #####################################
    # SET THE FOLLOWING PARAMETERS
    # MNIST DATABASE
    # total number of samples: 70000 (each is 28x28)
    number_of_train_samples = 10000 #65000
    number_of_test_samples = 1000 #70000 - number_of_train_samples
    # END OF PARAMETERS SETTING
    if (number_of_train_samples + number_of_test_samples) > len(mnist.data):
        print("ERROR, too much samples set!")
    #####################################

    from sklearn.utils import shuffle
    # TODO - is this enough or should I use my own funtion?
    mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
    #shuffle_data_and_target(mnist.data, mnist.target)

    train_data = mnist.data[:number_of_train_samples]
    train_target = mnist.target[:number_of_train_samples]
    test_data = mnist.data[number_of_train_samples:number_of_train_samples+number_of_test_samples]
    test_target = mnist.target[number_of_train_samples:number_of_train_samples+number_of_test_samples]

    test_dataset(mnist.data.shape[1], train_data, train_target, test_data, test_target)
