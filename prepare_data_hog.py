import skimage.data
import skimage.feature
import skimage
import matplotlib.pyplot as plt
import glob
import random
import pickle
import HOG_modified

#####################################
# SET THE FOLLOWING PARAMETERS
# INRIA DATABASE FOR HOG (64x128)
width = 64
height = 128
region_size = 16
number_of_positive_samples = 200
number_of_positive_tests = 50
number_of_negative_samples = 200
number_of_negative_tests = 50
files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"
positive_samples_directory = files_directory + "70X134H96\\Test\\pos\\"
negative_train_samples_directory = files_directory + "\\Train\\neg\\"
negative_test_samples_directory = files_directory + "\\Test\\neg\\"
# END OF PARAMETERS SETTING
#####################################

def show_images(original, hog):
    fig, (ax_image_original, ax_image_hog) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax_image_original.axis('off')
    ax_image_original.imshow(original, cmap=plt.cm.gray)
    ax_image_original.set_title('Input image')
    ax_image_original.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = skimage.exposure.rescale_intensity(hog, in_range=(0, 0.02))

    ax_image_hog.axis('off')
    ax_image_hog.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax_image_hog.set_title('Histogram of Oriented Gradients')
    ax_image_hog.set_adjustable('box-forced')
    plt.draw()

    plt.pause(0.0001)
    plt.waitforbuttonpress()

    plt.close()


def get_description_of_image_from_file(filename, flag_use_part_of_image=False, show=False):
    print(filename)

    image_from_file = skimage.data.imread(filename, as_grey=True)

    nrows, ncols = image_from_file.shape

    if flag_use_part_of_image:
        # choose left upper corner
        if nrows > height and ncols > width:
            x = random.randint(0, nrows - 1 - height)
            y = random.randint(0, ncols - 1 - width)
            # print x, y
            part_of_image = image_from_file[x:x + height, y:y + width]
            # print part_of_image
            image_from_file = part_of_image
    else:
        # crop the image to 64 x 128 pixel size
        image_from_file = skimage.util.crop(image_from_file, 3)

    # TODO - should we add global image normalisation?
    # http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=hog#skimage.feature.hog
    feature_vector, hog_image = skimage.feature.hog(image_from_file, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True)

    #feature_vector = HOG_modified.hog(image_from_file)

    if len(feature_vector) != 3780:
        print("Wrong feature vector size for specified HOG parameters. Should be: 3780, while it is: " + str(len(feature_vector)))

    if show:
        show_images(image_from_file, hog_image)

    return feature_vector


if __name__ == "__main__":
    train_data_positive = []
    test_data_positive = []
    train_data_negative = []
    test_data_negative = []

    # prepare positive examples
    png_filenames = glob.glob(positive_samples_directory + "*.png")
    for filename in png_filenames[:number_of_positive_samples]:
        data = get_description_of_image_from_file(filename, show=False)
        train_data_positive.append(data)

    for filename in png_filenames[number_of_positive_samples:(number_of_positive_samples+number_of_positive_tests)]:
        data = get_description_of_image_from_file(filename, show=False)
        test_data_positive.append(data)

    # prepare negative examples
    png_filenames = glob.glob(negative_train_samples_directory + "*.png") + glob.glob(negative_train_samples_directory + "*.jpg")
    for filename in png_filenames[:number_of_negative_samples]:
        data = get_description_of_image_from_file(filename, flag_use_part_of_image=True, show=False)
        train_data_negative.append(data)

    png_filenames = glob.glob(negative_test_samples_directory + "*.png") + glob.glob(negative_test_samples_directory + "*.jpg")
    for filename in png_filenames[:number_of_negative_tests]:
        data = get_description_of_image_from_file(filename, flag_use_part_of_image=True, show=False)
        test_data_negative.append(data)

    print(len(train_data_positive))
    with open("data\\positive_data", "wb") as f:
        pickle.dump(train_data_positive, f)

    print(len(test_data_positive))
    with open("data\\test_positive_data", "wb") as f:
        pickle.dump(test_data_positive, f)

    print(len(train_data_negative))
    with open("data\\negative_data", "wb") as f:
        pickle.dump(train_data_negative, f)

    print(len(test_data_negative))
    with open("data\\test_negative_data", "wb") as f:
        pickle.dump(test_data_negative, f)
