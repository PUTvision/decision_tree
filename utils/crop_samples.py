import glob
import os
import random
import skimage.io


def load_images_and_randomly_sample(input_directory, output_directory):
    samples_filenames = glob.glob(input_directory + "*.png") + glob.glob(input_directory + "*.jpg")

    for filename in samples_filenames:
        image_from_file = skimage.io.imread(filename, as_grey=True)

        nrows, ncols = image_from_file.shape

        # choose left upper corner
        if nrows > height and ncols > width:
            x = random.randint(0, nrows - 1 - height)
            y = random.randint(0, ncols - 1 - width)

            part_of_image = image_from_file[x:x + height, y:y + width]

            image_from_file = part_of_image

        skimage.io.imsave(output_directory + "cropped_" + os.path.basename(filename)[:-4] + ".png", image_from_file)


if __name__ == "__main__":

    #####################################
    # SET THE FOLLOWING PARAMETERS
    # INRIA DATABASE FOR HOG (64x128)
    width = 64
    height = 128
    files_directory = "F:\\Amin\\Desktop\\INRIAPerson\\"
    # END OF PARAMETERS SETTING
    #####################################

    #load_images_and_randomly_sample(files_directory + "Train\\neg\\", files_directory + "70X134H96\\Train\\neg\\")
    load_images_and_randomly_sample(files_directory + "Test\\neg\\", files_directory + "70X134H96\\Test\\neg\\")
