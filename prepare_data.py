__author__ = 'Amin'

import numpy as np
import matplotlib.pyplot as plt

from skimage import exposure
from skimage import data
from skimage.feature import local_binary_pattern
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.viewer import ImageViewer
from skimage.io import imshow
from skimage.io import show

import glob

import time

import random

from sklearn.preprocessing import normalize

import pickle


# this is test function for checking different ways of calculating the histogram
def calculate_histogram(image, number_of_bins):
    # different ways to calculate histogram
    histogram_from_np, bins_from_np = np.histogram(image, bins=number_of_bins)
    histogram_from_scimage, bins_from_scimage = exposure.histogram(image, nbins=number_of_bins)
    # important - the image data has to be processed by ravel method
    histogram_from_matplotlib, bins_from_matplotlib, patches_from_matplotlib = plt.hist(image.ravel(), bins=number_of_bins, normed=False)

    # all of these methods return same result
    print histogram_from_np
    print histogram_from_scimage
    print histogram_from_matplotlib

    # how to normalize
    #normalized_histogram = normalize(img_as_float(histogram_from_np[:, np.newaxis]), axis=0).ravel()
    # this can also be done as a parameter for matplotlib hist method


def show_histograms(original_image, lbp_image, nr_of_image_bins, nr_of_lbp_bins):
    # showing the results in the window
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

    #time.sleep(0.05)
    #raw_input("Press Enter to continue...")


def get_description_of_image_from_file(filename, flag_use_part_of_image=False, show=False):
    print filename

    width = 96
    height = 160
    region_size = 16

    image_from_file = img_as_ubyte(data.imread(filename, as_grey=True))

    if flag_use_part_of_image:
        # we have to extract the patch of size 96x160 so negative samples will be the same as positive

        # choose left upper corner
        nrows, ncols = image_from_file.shape
        #print nrows, ncols
        x = random.randint(0, nrows-1-height)
        y = random.randint(0, ncols-1-width)
        #print x, y
        part_of_image = image_from_file[x:x+height, y:y+width]
        #print part_of_image
        #viewer = ImageViewer(part_of_image)
        #viewer.show()
        image_from_file = part_of_image

    lbp_image = local_binary_pattern(image_from_file, n_points, radius, METHOD)

    # example function for calculating histograms in different ways
    #calculate_histogram(lbp_image, number_of_lbp_bins)

    image_description = np.array([])
    for i in xrange(0, width, region_size):
        for j in xrange(0, height, region_size):
            #print "i: ", i, ", j: ", j
            current_image_region = lbp_image[j:j+region_size, i:i+region_size]
            # np.histogram function is much faster than plt.hist
            lbp_histogram, lbp_bins = np.histogram(current_image_region, bins=number_of_lbp_bins)
            image_description = np.concatenate([image_description, lbp_histogram])

    # function for showing the original and lbp images and theirs histograms
    if show == True:
        show_histograms(image_from_file, lbp_image, number_of_image_bins, number_of_lbp_bins)

    #print len(lbp_histogram)
    #print lbp_histogram

    return image_description

if __name__ == "__main__":
    # settings for LBP
    radius = 1
    n_points = 8 * radius
    METHOD = "nri_uniform"  # 59 different types
    # "uniform"  # 10 different types
    # "default"  # 256 different types

    # settings for histograms
    number_of_image_bins = 256
    number_of_lbp_bins = 59

    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
    plt.gray()
    plt.ion()
    plt.show()

    number_of_positive_samples = 1200
    number_of_positive_tests = 400

    number_of_negative_samples = 1200
    number_of_negative_tests = 400

    train_histogram_positive = []
    test_histogram_positive = []
    train_histogram_negative = []
    test_histogram_negative = []

    files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"

    # version 1
    # load each image in directory
    #png_filenames = glob.glob(files_directory + "Train\\pos\\*.png")
    png_filenames = glob.glob(files_directory + "96X160H96\\Train\\pos\\*.png")
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
    png_filenames = glob.glob(files_directory + "\\Train\\neg\\*.png") + glob.glob(files_directory + "\\Train\\neg\\*.jpg")
    for filename in png_filenames[:number_of_negative_samples]:

        hist = get_description_of_image_from_file(filename, flag_use_part_of_image=True, show=False)
        train_histogram_negative.append(hist)

    png_filenames = glob.glob(files_directory + "\\Test\\neg\\*.png") + glob.glob(files_directory + "\\Test\\neg\\*.jpg")
    for filename in png_filenames[:number_of_negative_tests]:

        hist = get_description_of_image_from_file(filename, flag_use_part_of_image=True, show=False)
        test_histogram_negative.append(hist)

    print len(train_histogram_positive)
    with open("positive_histograms", "wb") as f:
        pickle.dump(train_histogram_positive, f)

    print len(test_histogram_positive)
    with open("test_positive_histograms", "wb") as f:
        pickle.dump(test_histogram_positive, f)

    print len(train_histogram_negative)
    with open("negative_histograms", "wb") as f:
        pickle.dump(train_histogram_negative, f)

    print len(test_histogram_negative)
    with open("test_negative_histograms", "wb") as f:
        pickle.dump(test_histogram_negative, f)

    #original_image = img_as_ubyte(data.imread("crop_000001a.png", as_grey=True))
    #original_image = img_as_ubyte(data.imread("F:\\Amin\\Desktop\\INRIAPerson\\Train\\pos\\person_and_bike_209.png", as_grey=True))
    #original_image = img_as_ubyte(rgb2gray(data.camera()))
    #original_image = img_as_ubyte(rgb2gray(data.lena()))

    #plt.bar(bins_from_np[:-1], normalized_histogram, width=1)
    #plt.xlim(min(bins_from_np), max(bins_from_np))
