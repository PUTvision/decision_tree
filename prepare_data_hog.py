import skimage.data
import skimage.feature
import skimage
import matplotlib.pyplot as plt
import pickle
import HOG_modified

#####################################
# SET THE FOLLOWING PARAMETERS
# INRIA DATABASE FOR HOG (64x128)
width = 64
height = 128
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


def HOG_function(filename, flag_use_skimage_version):
    print(filename)

    image_from_file = skimage.data.imread(filename, as_grey=True)

    nrows, ncols = image_from_file.shape
    if nrows == 134 and ncols == 70:
        # crop the image to 64 x 128 pixel size (crop from all sides by 3 pixels)
        image_from_file = skimage.util.crop(image_from_file, 3)
    elif nrows > height and ncols > width:
        print("\n\nERROR, image of a wrong size!\n\n")

    if flag_use_skimage_version:
        feature_vector_original = skimage.feature.hog(image_from_file, orientations=9,
                                                      pixels_per_cell=(8, 8),
                                                      cells_per_block=(2, 2))
    else:
        feature_vector_original = HOG_modified.hog(image_from_file)

    if len(feature_vector_original) != 3780:
        print("Wrong feature vector size for specified HOG parameters. Should be: 3780)")
        print("Feature vector actual size: " + str(len(feature_vector_original)))

    return feature_vector_original


def load_filenames_process_and_save_results(filename, flag_use_skimage_version):
    # load samples filenames
    with open("data\\samples_filenames\\" + filename + "_filenames.pickle", "rb") as f:
        samples_filenames = pickle.load(f)

    data = []
    for sample_filename in samples_filenames:
        single_data = HOG_function(sample_filename, flag_use_skimage_version)
        data.append(single_data)

    print(len(data))
    if flag_use_skimage_version:
        output_filename_modifier = ""
    else:
        output_filename_modifier = "_modified"
    with open("data\\" + filename + output_filename_modifier + ".pickle", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":

    # load_filenames_process_and_save_results("positive_train_samples", True)
    # load_filenames_process_and_save_results("positive_test_samples", True)
    # load_filenames_process_and_save_results("negative_train_samples", True)
    # load_filenames_process_and_save_results("negative_test_samples", True)

    load_filenames_process_and_save_results("positive_train_samples", False)
    load_filenames_process_and_save_results("positive_test_samples", False)
    load_filenames_process_and_save_results("negative_train_samples", False)
    load_filenames_process_and_save_results("negative_test_samples", False)
