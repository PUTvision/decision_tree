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

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

# settings for histograms
number_of_image_bins = 255
number_of_lbp_bins = 25

fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
plt.gray()
plt.ion()
plt.show()

train_histogram_positive = []
train_histogram_negative = []

files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"

# version 1
# load each image in directory
#png_filenames = glob.glob(files_directory + "Train\\pos\\*.png")
png_filenames = glob.glob(files_directory + "96X160H96\\Train\\pos\\*.png")
for filename in png_filenames[:10]:
    print filename

# version 2
# load each image listed in specified text file
#f = open(files_directory + "Train\\pos.lst", "r")
#lines_from_file = f.read().split("\n")
#image_from_file = None
#for line in lines_from_file[:5]:
    #print files_directory + line.replace("/", "\\") + "\n"
    #filename = files_directory + line.replace("/", "\\")

    image_from_file = img_as_ubyte(data.imread(filename, as_grey=True))
    lbp_image = local_binary_pattern(image_from_file, n_points, radius, METHOD)

    #calculate_histogram(lbp_image, number_of_lbp_bins)

    # showing the results in the window
    image_set = (image_from_file, lbp_image)
    bins_set = (number_of_image_bins, number_of_lbp_bins)

    lbp_histogram, lbp_bins, lbp_patches = plt.hist(lbp_image.ravel(), bins=number_of_lbp_bins, normed=False)

    train_histogram_positive.append(lbp_histogram)

    for ax, image in zip(ax_img, image_set):
        ax.imshow(image)

    for ax, image, number_of_bins in zip(ax_hist, image_set, bins_set):
        ax.cla()
        ax.hist(image.ravel(), normed=False, bins=number_of_bins, range=(0, number_of_bins))

    plt.draw()

    plt.pause(0.0001)
    plt.waitforbuttonpress()

    #time.sleep(0.05)
    #raw_input("Press Enter to continue...")

# prepare negative examples
# version 1
# load each image in directory
#png_filenames = glob.glob(files_directory + "Train\\pos\\*.png")
png_filenames = glob.glob(files_directory + "\\Train\\neg\\*.png")
for filename in png_filenames[:10]:
    image_from_file = img_as_ubyte(data.imread(filename, as_grey=True))
    # we have to extract the patch of size 96x160 so negative samples will be the same as positive
    width = 96
    height = 160
    # choose left upper corner
    nrows, ncols = image_from_file.shape
    #print nrows, ncols
    x = random.randint(0, nrows-1-width)
    y = random.randint(0, ncols-1-height)
    #print x, y
    part_of_image = image_from_file[x:x+width, y:y+height]
    #print part_of_image
    #viewer = ImageViewer(part_of_image)
    #viewer.show()
    lbp_image = local_binary_pattern(image_from_file, n_points, radius, METHOD)

    lbp_histogram, lbp_bins, lbp_patches = plt.hist(lbp_image.ravel(), bins=number_of_lbp_bins, normed=False)

    train_histogram_negative.append(lbp_histogram)


print len(train_histogram_positive)
with open("positive_histograms", "wb") as f:
    pickle.dump(train_histogram_positive, f)


print len(train_histogram_negative)
with open("negative_histograms", "wb") as f:
    pickle.dump(train_histogram_negative, f)


#with open("positive_histograms", "rb") as f:
#    train_histogram_positive = pickle.load(f)

#original_image = img_as_ubyte(data.imread("crop_000001a.png", as_grey=True))
#original_image = img_as_ubyte(data.imread("F:\\Amin\\Desktop\\INRIAPerson\\Train\\pos\\person_and_bike_209.png", as_grey=True))
#original_image = img_as_ubyte(rgb2gray(data.camera()))
#original_image = img_as_ubyte(rgb2gray(data.lena()))

#plt.bar(bins_from_np[:-1], normalized_histogram, width=1)
#plt.xlim(min(bins_from_np), max(bins_from_np))
