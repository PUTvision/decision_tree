__author__ = 'Amin'

import glob
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import skimage

from LBP import MF_lbp


def show_histograms(original_image, lbp_image, nr_of_image_bins, nr_of_lbp_bins):
    # show the results in the window and wait for the key to be pressed
    image_set = (original_image, lbp_image)
    bins_set = (nr_of_image_bins, nr_of_lbp_bins)

    for ax, image in zip(ax_img, image_set):
        ax.imshow(image)

    for ax, image, number_of_bins in zip(ax_hist, image_set, bins_set):
        ax.cla()
        ax.hist(image.ravel(), normed=False, bins=number_of_bins)

    plt.draw()

    plt.pause(0.0001)
    plt.waitforbuttonpress()


def get_description_of_image_from_file(filename, flag_use_part_of_image=False, show=False):
    print(filename)

    image_from_file = skimage.img_as_ubyte(skimage.data.imread(filename, as_grey=True))

    nrows, ncols = image_from_file.shape
    #print nrows, ncols

    if flag_use_part_of_image:
        # we have to extract the patch of size 96x160 so negative samples will be the same as positive

        # choose left upper corner
        if nrows > height and ncols > width:
            x = random.randint(0, nrows-1-height)
            y = random.randint(0, ncols-1-width)
            #print x, y
            part_of_image = image_from_file[x:x+height, y:y+width]
            #print part_of_image
            #viewer = ImageViewer(part_of_image)
            #viewer.show()
            image_from_file = part_of_image

    if nrows < height or ncols < width:
        # if the loaded image is smaller than the defined region duplicate pixels from the border
        vertical_difference = height - nrows
        if vertical_difference % 2 == 1:
            number_of_pixels_to_add_at_the_top = vertical_difference // 2 + 1
            number_of_pixels_to_add_at_the_bottom = vertical_difference // 2
        else:
            number_of_pixels_to_add_at_the_top = vertical_difference // 2
            number_of_pixels_to_add_at_the_bottom = vertical_difference // 2

        horizonatl_difference = width - ncols
        if vertical_difference % 2 == 1:
            number_of_pixels_to_add_on_the_left = horizonatl_difference // 2 + 1
            number_of_pixels_to_add_on_the_right = horizonatl_difference // 2
        else:
            number_of_pixels_to_add_on_the_left = horizonatl_difference // 2
            number_of_pixels_to_add_on_the_right = horizonatl_difference // 2

        part_of_image = np.lib.pad(image_from_file, (
            (number_of_pixels_to_add_at_the_top, number_of_pixels_to_add_at_the_bottom),
            (number_of_pixels_to_add_on_the_left, number_of_pixels_to_add_on_the_right)
        ), 'edge')

        image_from_file = part_of_image

    #lbp_image = local_binary_pattern(image_from_file, n_points, radius, METHOD)

    lbp_calculator = MF_lbp.MF_lbp(use_test_version=False)
    lbp_image = lbp_calculator.calc_nrulbp_3x3(image_from_file)

    # example function for calculating histograms in different ways
    #calculate_histogram(lbp_image, number_of_lbp_bins)

    image_description = np.array([])
    for i in range(0, width, region_size):
        for j in range(0, height, region_size):
            #print "i: ", i, ", j: ", j
            current_image_region = lbp_image[j:j+region_size, i:i+region_size]
            # np.histogram function is much faster than plt.hist
            lbp_histogram, lbp_bins = np.histogram(current_image_region, range=(0, 29), bins=number_of_lbp_bins)
            image_description = np.concatenate([image_description, lbp_histogram])

    # function for showing the original and lbp images and theirs histograms
    if show:
        show_histograms(image_from_file, lbp_image, number_of_image_bins, number_of_lbp_bins)

    #print len(lbp_histogram)
    #print lbp_histogram

    return image_description


def generate_image_file_for_vhdl_testbench(tb_image_filename, tb_data_filename):
    image_from_file = skimage.img_as_ubyte(skimage.data.imread(files_directory + tb_image_filename, as_grey=True))
    np.savetxt("data\\" + tb_data_filename, image_from_file, fmt="%d", delimiter="\n")

    hist = get_description_of_image_from_file(files_directory + tb_image_filename, show=True)

    np.savetxt("data\\hist_positive.csv", hist, fmt="%d", delimiter=",")

if __name__ == "__main__":

    #####################################
    # SET THE FOLLOWING PARAMETERS

    # INRIA DATABASE FOR RUC 2015, CORES 2015
    # width = 96
    # height = 160
    # region_size = 16
    # number_of_positive_samples = 1200
    # number_of_positive_tests = 400
    # number_of_negative_samples = 1200
    # number_of_negative_tests = 400
    # files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"
    # positive_samples_directory = files_directory + "96X160H96\\Train\\pos\\"
    # negative_train_samples_directory = files_directory + "\\Train\\neg\\"
    # negative_test_samples_directory = files_directory + "\\Test\\neg\\"

    # # TEST SETTINGS
    # width = 25
    # height = 25
    # region_size = 5
    # number_of_positive_samples = 1
    # number_of_positive_tests = 1
    # number_of_negative_samples = 2
    # number_of_negative_tests = 2
    # files_directory = "F:\\Amin\\Desktop\\sample_database\\"
    # positive_samples_directory = files_directory + "25x25\\Train\\positive\\"
    # negative_train_samples_directory = files_directory + "25x25\\Train\\negative\\"
    # negative_test_samples_directory = files_directory + "25x25\\Test\\negative\\"

    # CVC Virtual Pedestrian
    width = 50
    height = 100
    region_size = 5
    number_of_positive_samples = 500
    number_of_positive_tests = 500
    number_of_negative_samples = 1000
    number_of_negative_tests = 1000
    files_directory = "F:\\Amin\\Desktop\\CVC-Virtual-Pedestrian\\train\\"
    positive_samples_directory = files_directory + "pedestrians\\"
    negative_train_samples_directory = files_directory + "background-frames\\"
    negative_test_samples_directory = files_directory + "background-frames\\"

    # END OF PARAMETERS SETTING
    #####################################

    # settings for LBP
    radius = 1
    n_points = 8 * radius
    METHOD = "nri_uniform"  # 59 different types
    # "uniform"  # 10 different types
    # "default"  # 256 different types

    # settings for histograms
    number_of_image_bins = 256
    # this depends on which LBP method is used - the one from scikit-image generate 59 classes, \
    # while our own implementation generate 30
    number_of_lbp_bins = 30#59

    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
    plt.gray()
    plt.ion()
    plt.show()

    train_histogram_positive = []
    test_histogram_positive = []
    train_histogram_negative = []
    test_histogram_negative = []

    #generate_image_file_for_vhdl_testbench("data\\vhdl_tb_3_pixels.png", "tb_image_data_3_pixels.txt")

    # version 1
    # load each image in directory
    png_filenames = glob.glob(positive_samples_directory + "*.png")
    for filename in png_filenames[:number_of_positive_samples]:

    # version 2
    # load each image listed in specified text file
    #f = open(files_directory + "Train\\pos.lst", "r")
    #lines_from_file = f.read().split("\n")
    #for line in lines_from_file[:5]:
        #filename = files_directory + line.replace("/", "\\")

        hist = get_description_of_image_from_file(filename, show=False)
        train_histogram_positive.append(hist)

    for filename in png_filenames[number_of_positive_samples:(number_of_positive_samples+number_of_positive_tests)]:

        hist = get_description_of_image_from_file(filename, show=False)
        test_histogram_positive.append(hist)

    # prepare negative examples
    png_filenames = glob.glob(negative_train_samples_directory + "*.png") + glob.glob(negative_train_samples_directory + "*.jpg")
    for filename in png_filenames[:number_of_negative_samples]:

        hist = get_description_of_image_from_file(filename, flag_use_part_of_image=True, show=False)
        train_histogram_negative.append(hist)

    png_filenames = glob.glob(negative_test_samples_directory + "*.png") + glob.glob(negative_test_samples_directory + "*.jpg")
    for filename in png_filenames[:number_of_negative_tests]:

        hist = get_description_of_image_from_file(filename, flag_use_part_of_image=True, show=False)
        test_histogram_negative.append(hist)

    print(len(train_histogram_positive))
    with open("data\\positive_histograms", "wb") as f:
        pickle.dump(train_histogram_positive, f)

    print(len(test_histogram_positive))
    with open("data\\test_positive_histograms", "wb") as f:
        pickle.dump(test_histogram_positive, f)

    print(len(train_histogram_negative))
    with open("data\\negative_histograms", "wb") as f:
        pickle.dump(train_histogram_negative, f)

    print(len(test_histogram_negative))
    with open("data\\test_negative_histograms", "wb") as f:
        pickle.dump(test_histogram_negative, f)

    #original_image = img_as_ubyte(data.imread("crop_000001a.png", as_grey=True))
    #original_image = img_as_ubyte(data.imread("F:\\Amin\\Desktop\\INRIAPerson\\Train\\pos\\person_and_bike_209.png", as_grey=True))
    #original_image = img_as_ubyte(rgb2gray(data.camera()))
    #original_image = img_as_ubyte(rgb2gray(data.lena()))

    #plt.bar(bins_from_np[:-1], normalized_histogram, width=1)
    #plt.xlim(min(bins_from_np), max(bins_from_np))
