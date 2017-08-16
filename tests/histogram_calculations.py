import numpy as np
import skimage
import matplotlib.pyplot as plt
import sklearn.preprocessing


# this is a test function for checking different ways of calculating the histogram
def calculate_histogram(image, number_of_bins):
    # different ways to calculate histogram
    histogram_from_np, bins_from_np = np.histogram(image, bins=number_of_bins)
    histogram_from_scimage, bins_from_scimage = skimage.exposure.histogram(image, nbins=number_of_bins)
    # important - the image data has to be processed by ravel method
    histogram_from_matplotlib, bins_from_matplotlib, patches_from_matplotlib = plt.hist(image.ravel(), bins=number_of_bins, normed=False)

    # all of these methods should return same result
    print("histogram_from_np: \n" + str(histogram_from_np))
    print("histogram_from_scimage: \n" + str(histogram_from_scimage))
    print("histogram_from_matplotlib: \n" + str(histogram_from_matplotlib))

    # how to normalize
    #normalized_histogram = sklearn.preprocessing.normalize(skimage.img_as_float(histogram_from_np[:, np.newaxis]), axis=0).ravel()
    # this can also be done as a parameter for matplotlib hist method

if __name__ == "__main__":
    from skimage import data
    img = data.load('brick.png')
    calculate_histogram(img, 9)
