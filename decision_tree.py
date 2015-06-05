__author__ = 'Amin'

import numpy as np
import matplotlib.pyplot as plt

# classification
# training data
training_data = [[0, 0], [1, 1]]
class_labels = [0, 1]

# decision tree
# http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree

tree_classifier = tree.DecisionTreeClassifier()
tree_classifier = tree_classifier.fit(training_data, class_labels)
print tree_classifier.predict([0.51, 0.51])

# forest of randomized trees
# http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=10)
forest_classifier = forest_classifier.fit(training_data, class_labels)
print forest_classifier.predict([0.51, 0.51])

from skimage import exposure
from skimage import data
from skimage.feature import local_binary_pattern
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage import img_as_ubyte

import glob

import time

from sklearn.preprocessing import normalize


def calculate_histogram(image, number_of_bins):
    # different ways to calculate histogram
    histogram_from_np, bins_from_np = np.histogram(image, bins=number_of_bins)
    histogram_from_scimage, bins_from_scimage = exposure.histogram(image, nbins=number_of_bins)

    print histogram_from_np
    print histogram_from_scimage

    # how to normalize
    #normalized_histogram = normalize(img_as_float(histogram_from_np[:, np.newaxis]), axis=0).ravel()

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

# settings for histograms
number_of_image_bins = 255
number_of_lbp_bins = 25

# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
plt.gray()
plt.ion()
plt.show()

files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"

# version 1
# load each image in directory
png_filenames = glob.glob(files_directory + "Train\\pos\\*.png")
for filename in png_filenames[:5]:
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

    calculate_histogram(lbp_image, number_of_lbp_bins)

    image_set = (image_from_file, lbp_image)
    bins_set = (number_of_image_bins, number_of_lbp_bins)

    for ax, image in zip(ax_img, image_set):
        ax.imshow(image)

    for ax, image, number_of_bins in zip(ax_hist, image_set, bins_set):
        ax.cla()
        n, bins, patches = ax.hist(image.ravel(), normed=False, bins=number_of_bins, range=(0, number_of_bins))
        print n, bins, patches, "\n"

    plt.draw()

    plt.pause(0.0001)
    plt.waitforbuttonpress()

    #time.sleep(0.05)
    #raw_input("Press Enter to continue...")


#original_image = img_as_ubyte(data.imread("crop_000001a.png", as_grey=True))
#original_image = img_as_ubyte(data.imread("F:\\Amin\\Desktop\\INRIAPerson\\Train\\pos\\person_and_bike_209.png", as_grey=True))
#original_image = img_as_ubyte(rgb2gray(data.camera()))
#original_image = img_as_ubyte(rgb2gray(data.lena()))













#plt.bar(bins_from_np[:-1], normalized_histogram, width=1)
#plt.xlim(min(bins_from_np), max(bins_from_np))
