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


#encoded_lbp_lut = [29] * 256
encoded_lbp_lut = [0] * 256
encoded_lbp_lut[0], encoded_lbp_lut[255] = 0, 0
# encoded_lbp_lut[1], encoded_lbp_lut[254] = 1, 1
# encoded_lbp_lut[2], encoded_lbp_lut[253] = 2, 2
# encoded_lbp_lut[3], encoded_lbp_lut[252] = 3, 3
# encoded_lbp_lut[4], encoded_lbp_lut[251] = 4, 4
# encoded_lbp_lut[6], encoded_lbp_lut[249] = 5, 5
# encoded_lbp_lut[7], encoded_lbp_lut[248] = 6, 6
# encoded_lbp_lut[8], encoded_lbp_lut[247] = 7, 7
# encoded_lbp_lut[12], encoded_lbp_lut[243] = 8, 8
# encoded_lbp_lut[14], encoded_lbp_lut[241] = 9, 9
# encoded_lbp_lut[15], encoded_lbp_lut[240] = 10, 10
# #encoded_lbp_lut[16], encoded_lbp_lut[239] = 11, 11
# encoded_lbp_lut[24], encoded_lbp_lut[231] = 12, 12
# encoded_lbp_lut[28], encoded_lbp_lut[227] = 13, 13
# encoded_lbp_lut[30], encoded_lbp_lut[225] = 14, 14
# encoded_lbp_lut[31], encoded_lbp_lut[224] = 15, 15
# encoded_lbp_lut[32], encoded_lbp_lut[223] = 16, 16
# encoded_lbp_lut[48], encoded_lbp_lut[207] = 17, 17
# encoded_lbp_lut[56], encoded_lbp_lut[199] = 18, 18
# encoded_lbp_lut[60], encoded_lbp_lut[195] = 19, 19
# encoded_lbp_lut[62], encoded_lbp_lut[193] = 20, 20
# encoded_lbp_lut[63], encoded_lbp_lut[192] = 21, 21
# encoded_lbp_lut[64], encoded_lbp_lut[191] = 22, 22
# encoded_lbp_lut[96], encoded_lbp_lut[159] = 23, 23
# encoded_lbp_lut[112], encoded_lbp_lut[243] = 24, 24
# encoded_lbp_lut[120], encoded_lbp_lut[135] = 25, 25
# encoded_lbp_lut[124], encoded_lbp_lut[131] = 26, 26
# encoded_lbp_lut[126], encoded_lbp_lut[129] = 27, 27
# encoded_lbp_lut[127], encoded_lbp_lut[128] = 28, 28
encoded_lbp_lut[16] = 11


def nrulbp_3x3(image):

    output_shape = (image.shape[0], image.shape[1])
    nrulbp_3x3_image = np.zeros(output_shape, dtype=np.double)

    rows = image.shape[0]
    cols = image.shape[1]

    for r in xrange(1, rows-1):
        for c in xrange(1, cols-1):
            central_pixel = int(image[r, c])
            raw_lbp_descriptor = 0

            if int(image[r-1, c-1]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 0)
            if int(image[r-1, c]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 1)
            if int(image[r-1, c+1]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 2)
            if int(image[r, c-1]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 3)
            if int(image[r, c+1]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 4)
            if int(image[r+1, c-1]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 5)
            if int(image[r+1, c]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 6)
            if int(image[r+1, c+1]) - central_pixel > 0:
                raw_lbp_descriptor += 1 * pow(2, 7)

            nrulbp_3x3_image[r, c] = encoded_lbp_lut[raw_lbp_descriptor]

    return np.asarray(nrulbp_3x3_image)


def get_description_of_image_from_file(filename, flag_use_part_of_image=False, show=False):
    print filename

    image_from_file = img_as_ubyte(data.imread(filename, as_grey=True))

    if flag_use_part_of_image:
        # we have to extract the patch of size 96x160 so negative samples will be the same as positive

        # choose left upper corner
        nrows, ncols = image_from_file.shape
        #print nrows, ncols

        if nrows > height and ncols > width:
            x = random.randint(0, nrows-1-height)
            y = random.randint(0, ncols-1-width)
            #print x, y
            part_of_image = image_from_file[x:x+height, y:y+width]
            #print part_of_image
            #viewer = ImageViewer(part_of_image)
            #viewer.show()
            image_from_file = part_of_image

    #lbp_image = local_binary_pattern(image_from_file, n_points, radius, METHOD)

    lbp_image = nrulbp_3x3(image_from_file)

    # example function for calculating histograms in different ways
    #calculate_histogram(lbp_image, number_of_lbp_bins)

    image_description = np.array([])
    for i in xrange(0, width, region_size):
        for j in xrange(0, height, region_size):
            #print "i: ", i, ", j: ", j
            current_image_region = lbp_image[j:j+region_size, i:i+region_size]
            # np.histogram function is much faster than plt.hist
            lbp_histogram, lbp_bins = np.histogram(current_image_region, range=(0, 29), bins=number_of_lbp_bins)
            image_description = np.concatenate([image_description, lbp_histogram])

    # function for showing the original and lbp images and theirs histograms
    if show == True:
        show_histograms(image_from_file, lbp_image, number_of_image_bins, number_of_lbp_bins)

    #print len(lbp_histogram)
    #print lbp_histogram

    return image_description

if __name__ == "__main__":

    #####################################
    # SET FOLLOWING PARAMETERS

    number_of_positive_samples = 1#1200
    number_of_positive_tests = 1#400

    number_of_negative_samples = 2#1200
    number_of_negative_tests = 2#400

    #files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"
    files_directory = "F:\\Amin\\Desktop\\sample_database\\"

    #positive_samples_directory = files_directory + "Train\\pos\\"
    #positive_samples_directory = files_directory + "96X160H96\\Train\\pos\\"
    positive_samples_directory = files_directory + "25x25\\Train\\positive\\"

    #negative_samples_directory = files_directory + "\\Train\\neg\\"
    negative_train_samples_directory = files_directory + "25x25\\Train\\negative\\"

    #negative_test_samples_directory = files_directory + "\\Test\\neg\\"
    negative_test_samples_directory = files_directory + "25x25\\Test\\negative\\"

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
    number_of_lbp_bins = 30#59

    # settings for image region
    width = 25#96
    height = 25#160
    region_size = 5#16

    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
    plt.gray()
    plt.ion()
    plt.show()

    train_histogram_positive = []
    test_histogram_positive = []
    train_histogram_negative = []
    test_histogram_negative = []

    image_from_file = img_as_ubyte(data.imread(files_directory + "vhdl_tb.png", as_grey=True))
    np.savetxt("tb_image_data.txt", image_from_file, fmt="%d", delimiter="\n")

    hist = get_description_of_image_from_file(files_directory + "vhdl_tb.png", show=True)

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

        np.savetxt("hist_positive.csv", hist, fmt="%d", delimiter=",")

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
