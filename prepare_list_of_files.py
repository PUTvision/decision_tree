import glob
import random
import pickle

if __name__ == "__main__":
    #####################################
    # SET THE FOLLOWING PARAMETERS
    # INRIA DATABASE FOR HOG (64x128)
    # total number of positive samples: 1126
    number_of_positive_samples = 900
    number_of_positive_tests = 200
    # total number of negative train samples: 1218
    number_of_negative_samples = 1200
    # total number of negative test samples: 453
    number_of_negative_tests = 400
    files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"
    positive_samples_directory = files_directory + "70X134H96\\Test\\pos\\"
    negative_train_samples_directory = files_directory + "70X134H96\\Train\\neg\\"
    negative_test_samples_directory = files_directory + "70X134H96\\Test\\neg\\"
    # END OF PARAMETERS SETTING
    #####################################

    # prepare positive examples
    positive_samples_filenames = glob.glob(positive_samples_directory + "*.png")
    random.shuffle(positive_samples_filenames)
    print(len(positive_samples_filenames[:number_of_positive_samples]))
    with open("data\\samples_filenames\\positive_train_samples_filenames.pickle", "wb") as f:
        pickle.dump(positive_samples_filenames[:number_of_positive_samples], f)

    print(len(positive_samples_filenames[number_of_positive_samples:(number_of_positive_samples+number_of_positive_tests)]))
    with open("data\\samples_filenames\\positive_test_samples_filenames.pickle", "wb") as f:
        pickle.dump(positive_samples_filenames[number_of_positive_samples:(number_of_positive_samples+number_of_positive_tests)], f)

    # prepare negative train examples
    negative_train_samples_filenames = glob.glob(negative_train_samples_directory + "*.png") + glob.glob(negative_train_samples_directory + "*.jpg")
    random.shuffle(negative_train_samples_filenames)
    print(len(negative_train_samples_filenames[:number_of_negative_samples]))
    with open("data\\samples_filenames\\negative_train_samples_filenames.pickle", "wb") as f:
        pickle.dump(negative_train_samples_filenames[:number_of_negative_samples], f)

    # prepare negative train examples
    negative_test_samples_filenames = glob.glob(negative_test_samples_directory + "*.png") + glob.glob(negative_test_samples_directory + "*.jpg")
    random.shuffle(negative_test_samples_filenames)
    print(len(negative_test_samples_filenames[:number_of_negative_tests]))
    with open("data\\samples_filenames\\negative_test_samples_filenames.pickle", "wb") as f:
        pickle.dump(negative_test_samples_filenames[:number_of_negative_tests], f)
