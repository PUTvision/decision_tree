__author__ = 'Amin'

import numpy as np
import sklearn.preprocessing
import math
import matplotlib.pyplot as plt


class GradientDir:

    def __init__(self):
        pass

    def compare_implemenatation(self):
        test_size = 50

        pixel_descriptors_tangent = np.empty((test_size*2, test_size*2), dtype=object)
        pixel_descriptors_vhdl = np.empty((test_size*2, test_size*2), dtype=object)

        row_to_check = -30
        col_to_check = -17

        number_of_differences = 0

        for row in range(-test_size, test_size+1):
            for col in range(-test_size, test_size+1):
                gradient_x = row
                gradient_y = col

                gradient_norm = self._gradient_norm(gradient_x, gradient_y)

                if row == row_to_check and col == col_to_check:
                    print("Debug go!")

                pd = self._compute_and_combine_dir_and_demux_gradien(gradient_x, gradient_y, gradient_norm)
                pixel_descriptors_tangent[row][col] = np.asarray(pd.ravel().tolist())

                abs_gradient_x, abs_gradient_y, sign_flag = self._shuffle_signs(gradient_x, gradient_y)
                bins_0, bins_1 = self._compute_dir(abs_gradient_x, abs_gradient_y, sign_flag)
                gradient_dir = self._combine_dir(bins_0, bins_1)

                pixel_descriptor = self._gradient_demux(gradient_dir, gradient_norm)
                #pixel_descriptor = self._downto_fix(pixel_descriptor)
                pixel_descriptor = self._bin_mixing(pixel_descriptor)

                pd_diff = pd - pixel_descriptor
                if not np.all(pd_diff == 0):
                    number_of_differences += 1
                    print(pd)
                    print(pixel_descriptor)
                    #print("row: " + str(row) + ", col: " + str(col))
                    if gradient_x == 0 and gradient_y == 0:
                        angle = 0
                    # both the +inf and -inf are put to the same value, as either way it will go to the same bins
                    elif gradient_x == 0 and gradient_y != 0:
                        angle = -math.pi / 2  # also could be math.pi/2
                    else:
                        angle = math.atan(gradient_y / gradient_x)

                    angle_in_degrees = angle * 180 / math.pi

                    print("x: " + str(gradient_x) + ", y: " + str(gradient_y) + ", angle: " + str(angle_in_degrees))



                pixel_descriptors_vhdl[row][col] = np.asarray(pixel_descriptor)

        pixel_descriptors_difference = pixel_descriptors_tangent - pixel_descriptors_vhdl

        print("Number pf differences: " + str(number_of_differences))
        print("Done!")

    def _downto_fix(self, pixel_descriptor):
        import copy
        temp = copy.deepcopy(pixel_descriptor)
        for i in range(0, len(pixel_descriptor)):
            pixel_descriptor[i] = temp[len(pixel_descriptor)-1 - i]

        return pixel_descriptor

    def _bin_mixing(self, pixel_descriptor):
        import copy
        temp = copy.deepcopy(pixel_descriptor)
        pixel_descriptor[0] = temp[4]
        pixel_descriptor[1] = temp[5]
        pixel_descriptor[2] = temp[6]
        pixel_descriptor[3] = temp[7]
        pixel_descriptor[4] = temp[8]
        pixel_descriptor[5] = temp[0]
        pixel_descriptor[6] = temp[1]
        pixel_descriptor[7] = temp[2]
        pixel_descriptor[8] = temp[3]

        return pixel_descriptor

        return pixel_descriptor

    def compute(self, image: np.ndarray):

        pixel_descriptors = np.empty(image.shape, dtype=object)

        empty_pixel_descriptor = np.zeros(9)

        pixel_descriptors[0] = [empty_pixel_descriptor for item in pixel_descriptors[0]]
        pixel_descriptors[-1] = [empty_pixel_descriptor for item in pixel_descriptors[0]]
        pixel_descriptors[:, 0] = [empty_pixel_descriptor for item in pixel_descriptors[:, 0]]
        pixel_descriptors[:, -1] = [empty_pixel_descriptor for item in pixel_descriptors[:, -1]]

        for row in range(1, image.shape[0]-1):
            for col in range(1, image.shape[1]-1):
                gradient_x = image[row, col+1] - image[row, col-1]
                gradient_y = image[row+1, col] - image[row-1, col]
                #if gradient_x != 0:
                #    print("x: (" + str(row) + ", " + str(col) + "): " + str(gradient_x))
                #if gradient_y != 0:
                #    print("y: (" + str(row) + ", " + str(col) + "): " + str(gradient_y))

                abs_gradient_x, abs_gradient_y, sign_flag = self._shuffle_signs(gradient_x, gradient_y)
                #print("abs_gradient_x: " + str(abs_gradient_x) + ", abs_gradient_y: " + str(abs_gradient_y))

                bins_0, bins_1 = self._compute_dir(abs_gradient_x, abs_gradient_y, sign_flag)
                #print("bins_0: " + str(bin(bins_0)) + ", bins_1: " + str(bin(bins_1)))

                gradient_dir = self._combine_dir(bins_0, bins_1)
                #if gradient_dir != int(0x140):  # decimal: 320
                    #print("gradient_dir: (" + str(row) + ", " + str(col) + "): " + str(hex(gradient_dir) + ",\t\t\t" + str(bin(gradient_dir)) ))

                gradient_norm = self._gradient_norm(gradient_x, gradient_y)
                #if gradient_norm != 0:
                    #print("gradient_norm: (" + str(row) + ", " + str(col) + "): " + str(gradient_norm))

                pixel_descriptor = self._gradient_demux(gradient_dir, gradient_norm)
                pixel_descriptor = self._bin_mixing(pixel_descriptor)
                #print("pixel_descriptor: " + str(pixel_descriptor))

                ##pd = self._compute_and_combine_dir_and_demux_gradien(gradient_x, gradient_y, gradient_norm)
                ##pixel_descriptor = pd.ravel().tolist()

                pixel_descriptors[row][col] = np.asarray(pixel_descriptor)


        #print("pixel_descriptors: \n" + str(pixel_descriptors))

        descriptors = self._gradient_adder(pixel_descriptors)
        #print("descriptors: \n" + str(descriptors))
        #print("descriptors size: \n" + str(descriptors.shape))

        #for row in descriptors[:]:
        #    for value in row[:]:
        #        print(str(value) + ", ")
        #    print("\n")

        return descriptors

    def _shuffle_signs(self, gradient_x, gradient_y):
        abs_gradient_x = abs(gradient_x)
        abs_gradient_y = abs(gradient_y)

        if (gradient_x < 0 and gradient_y >= 0) or (gradient_x >= 0 and gradient_y <0):
            sign_flag = True
        else:
            sign_flag = False

        return abs_gradient_x, abs_gradient_y, sign_flag

    # TODO - zamień na arctan(y/x) - 0 - 180 stopni
    # TODO - i potem mając kąt umieścić wagę go do dwóch sąsiednich "binów"
    # TODO - liniowo podzielić
    # TODO - na koniec nie zapomnieć, żeby przemnożyć uzyskane wagi przez grad_norm

    def _compute_and_combine_dir_and_demux_gradien(self, gradient_x, gradient_y, gradient_norm):

        # TODO - think how to handle division by 0 properly
        if gradient_x == 0 and gradient_y == 0:
            angle = 0
        # both the +inf and -inf are put to the same value, as either way it will go to the same bins
        elif gradient_x == 0 and gradient_y != 0:
            angle = -math.pi/2  # also could be math.pi/2
        else:
            angle = math.atan(gradient_y/gradient_x)

        angle_in_degrees = angle*180/math.pi

        #if angle_in_degrees != 0:
            #print("x: " + str(gradient_x) + ", y: " + str(gradient_y) + ", angle: " + str(angle_in_degrees))

        # place the angle into appropriate bins
        bins_center_angle = []
        for i in range(-80, 100, 20):
            bins_center_angle.append(float(i))
        #print(bins_center_angle)

        bins_values = np.zeros([1, 9])
        #print(bins_values)

        # if angle_in_degrees <= -80:
        #     bins_values[0, 0] = 0.5
        #     bins_values[0, 1] = 0.5
        # elif angle_in_degrees <= -70:
        #     bins_values[0, 0] = 0.5
        #     bins_values[0, 1] = 0.5
        # elif angle_in_degrees <= -50:
        #     bins_values[0, 1] = 0.5
        #     bins_values[0, 2] = 0.5
        # elif angle_in_degrees <= -30:
        #     bins_values[0, 2] = 0.5
        #     bins_values[0, 3] = 0.5
        # elif angle_in_degrees <= -10:
        #     bins_values[0, 3] = 0.5
        #     bins_values[0, 4] = 0.5
        # elif angle_in_degrees <= 10:
        #     bins_values[0, 4] = 0.5
        #     bins_values[0, 5] = 0.5
        # elif angle_in_degrees <= 30:
        #     bins_values[0, 5] = 0.5
        #     bins_values[0, 6] = 0.5
        # elif angle_in_degrees <= 50:
        #     bins_values[0, 6] = 0.5
        #     bins_values[0, 7] = 0.5
        # elif angle_in_degrees <= 70:
        #     bins_values[0, 7] = 0.5
        #     bins_values[0, 8] = 0.5
        # else:
        #     bins_values[0, 8] = 0.5
        #     bins_values[0, 0] = 0.5

        # TODO - take care of first and last range (should they be connected?)
        # check if the angle is smaller than center point of first bin
        if angle_in_degrees < bins_center_angle[0]:
            distance_to_current = abs(angle_in_degrees - bins_center_angle[0])
            distance_to_next = 20 - distance_to_current
            bins_values[0, 0] = 0.5#distance_to_next/20
            bins_values[0, 1] = 0.5#distance_to_current/20
        elif angle_in_degrees > bins_center_angle[-1]:
            distance_to_current = abs(angle_in_degrees - bins_center_angle[-1])
            distance_to_next = 20 - distance_to_current
            bins_values[0, 0] = 0.5#distance_to_next/20
            bins_values[0, 8] = 0.5#distance_to_current/20
        else:
            for i, [current_center_angle, next_center_angle] in enumerate(zip(bins_center_angle[:-1], bins_center_angle[1:])):
                if angle_in_degrees == current_center_angle:
                    #print("Found exactly: " + str(current_center_angle))
                    #bins_values[0, i] = 1
                    bins_values[0, i] = 0.5
                    bins_values[0, i+1] = 0.5

                elif current_center_angle < angle_in_degrees < next_center_angle:
                    #print("Found: " + str(i) + ", current_center_angle: " + str(current_center_angle) + ", next_center_angle: " + str(next_center_angle))

                    distance_to_current = abs(angle_in_degrees - current_center_angle)
                    distance_to_next = abs(angle_in_degrees - next_center_angle)

                    bins_values[0, i] = 0.5#distance_to_next/20
                    bins_values[0, i+1] = 0.5#distance_to_current/20

        # multiply by grad_norm
        bins_values = bins_values * gradient_norm
        #if bins_values.any():   # check if all are not 0
            #print(bins_values)

        return bins_values

    def _compute_dir(self, abs_gradient_x, abs_gradient_y, sign_flag):
        # watch out for approximations of values - here we divide, while in VHDL some bits are removed, so:
        # I guess it is ok, cause // operator was used (no reminder)
        mul_coeff_0_0 = (abs_gradient_x * 372) // 1024
        mul_coeff_0_1 = (abs_gradient_x * 859) // 1024
        mul_coeff_0_2 = (abs_gradient_x * 1773) // 1024
        mul_coeff_0_3 = (abs_gradient_x * 5807) // 1024
        mul_coeff_1_0 = (abs_gradient_x * 180) // 1024
        mul_coeff_1_1 = (abs_gradient_x * 591) // 1024
        mul_coeff_1_2 = (abs_gradient_x * 1220) // 1024
        mul_coeff_1_3 = (abs_gradient_x * 2813) // 1024

        if abs_gradient_x == 0 and abs_gradient_y == 0:
            bins_0 = 0b000000001
            bins_1 = 0b100000000
        elif abs_gradient_x == 0 and abs_gradient_y != 0:
            bins_0 = 0b000010000
            bins_1 = 0b000010000
        else:
            # TODO - in VHDL it is possible to remove the comparison with 0
            if abs_gradient_y < mul_coeff_0_0:
                if not sign_flag:
                    bins_0 = 0b000000001
                else:
                    bins_0 = 0b100000000
            elif mul_coeff_0_0 <= abs_gradient_y < mul_coeff_0_1:
                if not sign_flag:
                    bins_0 = 0b000000010
                else:
                    bins_0 = 0b010000000
            elif mul_coeff_0_1 <= abs_gradient_y < mul_coeff_0_2:
                if not sign_flag:
                    bins_0 = 0b000000100
                else:
                    bins_0 = 0b001000000
            elif mul_coeff_0_2 <= abs_gradient_y < mul_coeff_0_3:
                if not sign_flag:
                    bins_0 = 0b000001000
                else:
                    bins_0 = 0b000100000
            else:  # abs_gradient_y >= mul_coeff_0_3
                bins_0 = 0b000010000

            if abs_gradient_y < mul_coeff_1_0:
                bins_1 = 0b100000000
            elif mul_coeff_1_0 <= abs_gradient_y < mul_coeff_1_1:
                if not sign_flag:
                    bins_1 = 0b000000001
                else:
                    bins_1 = 0b010000000
            elif mul_coeff_1_1 <= abs_gradient_y < mul_coeff_1_2:
                if not sign_flag:
                    bins_1 = 0b000000010
                else:
                    bins_1 = 0b001000000
            elif mul_coeff_1_2 <= abs_gradient_y < mul_coeff_1_3:
                if not sign_flag:
                    bins_1 = 0b000000100
                else:
                    bins_1 = 0b000100000
            else:  # abs_gradient_y >= mul_coeff_1_3
                if not sign_flag:
                    bins_1 = 0b000001000
                else:
                    bins_1 = 0b000010000

        return bins_0, bins_1

    def _combine_dir_old(self, bins_0, bins_1):
        bin_number = 0b000000000000000000  # bit length: 18 (elements are indexed 0 through 17)
        for i in range(8, -1, -1):
            bins_0_bit = bins_0 & (1 << i)
            bins_1_bit = bins_1 & (1 << i)
            v1 = ((bins_0_bit & bins_1_bit) >> i)
            v2 = ((bins_0_bit ^ bins_1_bit) >> i)
            bin_number |= (v1 << (2*i+1))
            bin_number |= (v2 << (2*i))

        return bin_number

    def _combine_dir(self, bins_0, bins_1):
        # zero the most significant bit
        additional_bins = 0b011111111 & bins_1
        #print(bin(additional_bins))
        # move it to the left by one
        additional_bins = additional_bins << 1
        #print(bin(additional_bins))
        # insert the last bit from bins_0 to the first position
        temp_bins = 0b100000000 & bins_1
        #print(bin(temp_bins))
        temp_bins = temp_bins >> 8
        #print(bin(temp_bins))

        additional_bins = additional_bins | temp_bins
        #print(bin(additional_bins))

        bin_number = bins_0 | bins_1 | additional_bins

        return bin_number

    def _gradient_norm(self, grad_x, grad_y):
        val1 = abs(grad_x)
        val2 = abs(grad_y)
        if val1 > val2:
            maximum = val1
            minimum = val2
        else:
            maximum = val2
            minimum = val1
        xterm = (maximum - np.floor(0.125 * maximum)) + np.floor(0.5 * minimum)
        if xterm > maximum:
            return xterm
        else:
            return maximum

    def _gradient_norm2(self, grad_x, grad_y):
        return math.sqrt(grad_x ** 2 + grad_y ** 2)

    def _gradient_demux_old(self, gradient_dir, gradient_norm):
        out = [0]*9

        if gradient_dir == 0b000000000000000101:
            out[0] = gradient_norm
            out[1] = gradient_norm
        elif gradient_dir == 0b000000000000010100:
            out[1] = gradient_norm
            out[2] = gradient_norm
        elif gradient_dir == 0b000000000001010000:
            out[2] = gradient_norm
            out[3] = gradient_norm
        elif gradient_dir == 0b000000000101000000:
            out[3] = gradient_norm
            out[4] = gradient_norm
        elif gradient_dir == 0b000000010100000000:
            out[4] = gradient_norm
            out[5] = gradient_norm
        elif gradient_dir == 0b000001010000000000:
            out[5] = gradient_norm
            out[6] = gradient_norm
        elif gradient_dir == 0b000101000000000000:
            out[6] = gradient_norm
            out[7] = gradient_norm
        elif gradient_dir == 0b010100000000000000:
            out[7] = gradient_norm
            out[8] = gradient_norm
        elif gradient_dir == 0b010000000000000001:
            out[8] = gradient_norm
            out[0] = gradient_norm
        elif gradient_dir == 0b000000000000000010:
            out[0] = gradient_norm * 2
        elif gradient_dir == 0b000000000000001000:
            out[1] = gradient_norm * 2
        elif gradient_dir == 0b000000000000100000:
            out[2] = gradient_norm * 2
        elif gradient_dir == 0b000000000010000000:
            out[3] = gradient_norm * 2
        elif gradient_dir == 0b000000001000000000:
            out[4] = gradient_norm * 2
        elif gradient_dir == 0b000000100000000000:
            out[5] = gradient_norm * 2
        elif gradient_dir == 0b000010000000000000:
            out[6] = gradient_norm * 2
        elif gradient_dir == 0b001000000000000000:
            out[7] = gradient_norm * 2
        elif gradient_dir == 0b100000000000000000:
            out[8] = gradient_norm * 2

        return out

    def _gradient_demux(self, gradient_dir, gradient_norm):
        out = [0]*9

        if gradient_dir == 0b000000011:
            out[0] = gradient_norm/2
            out[1] = gradient_norm/2
        elif gradient_dir == 0b000000110:
            out[1] = gradient_norm/2
            out[2] = gradient_norm/2
        elif gradient_dir == 0b000001100:
            out[2] = gradient_norm/2
            out[3] = gradient_norm/2
        elif gradient_dir == 0b000011000:
            out[3] = gradient_norm/2
            out[4] = gradient_norm/2
        elif gradient_dir == 0b000110000:
            out[4] = gradient_norm/2
            out[5] = gradient_norm/2
        elif gradient_dir == 0b001100000:
            out[5] = gradient_norm/2
            out[6] = gradient_norm/2
        elif gradient_dir == 0b011000000:
            out[6] = gradient_norm/2
            out[7] = gradient_norm/2
        elif gradient_dir == 0b110000000:
            out[7] = gradient_norm/2
            out[8] = gradient_norm/2
        elif gradient_dir == 0b100000001:
            out[8] = gradient_norm/2
            out[0] = gradient_norm/2
        else:
            print("error!")
            out[0] = gradient_norm/2
            out[1] = gradient_norm/2

        return out

    def _gradient_adder(self, grad_in: np.ndarray):
        # each cell has 9 values, this method computes the sum of all corresponding values

        final_descriptor = np.empty(grad_in.shape, dtype=object)
        #print("grad_in.shape: " + str(grad_in.shape))

        empty_final_descriptor = np.zeros(9)

        for row in range(0, final_descriptor.shape[0]):
            for col in range(0, final_descriptor.shape[1]):
                final_descriptor[row, col] = empty_final_descriptor

        row_start = 3
        row_end = 8 - (row_start + 1)

        col_start = 3
        col_end = 8 - (col_start + 1)

        #print("Starting row: " + str(row_start) + ", " + str(row_end) + "\n")
        #print("Starting col: " + str(col_start) + ", " + str(col_end) + "\n")

        for row in range(row_start, grad_in.shape[0]-row_end):
            for col in range(col_start, grad_in.shape[1]-col_end):

                subarray = np.asarray(grad_in[row-row_start:row+row_end+1, col-col_start:col+col_end+1])
                #print("subarray.shape: " + str(subarray.shape))

                sum = subarray.sum()
                #print("sum: " + str(sum))

                final_descriptor[row, col] = sum

        return final_descriptor


def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    gradients = GradientDir()
    # pixels_per_cell is integrated into _gradient_adder method of GradientDir and does operate on 8x8 windows
    descriptors = gradients.compute(image)

    # now the block 2x2, that overlaps themselves should be calculated
    # for the moment we are counting on the image provided to be 6 pixels wider (3 pixels on each side)
    # and 6 pixels higher (3 pixels on each size) than the required size og 64x128
    nrows, ncols = image.shape

    feature_vector = np.empty([0, 0])

    for row in range(3, nrows-8, 8):
        #print("Row: " + str(row))

        for col in range(3, ncols-8, 8):
            #print("Col: " + str(col))

            cell_feature_vector = [descriptors[row, col],
                                   descriptors[row, col + 8],
                                   descriptors[row + 8, col],
                                   descriptors[row + 8, col + 8]
                                   ]

            cell_feature_vector = np.reshape(cell_feature_vector, 36)
            cell_feature_vector = sklearn.preprocessing.normalize([cell_feature_vector], norm='l2').ravel()

            #print(cell_feature_vector)

            feature_vector = np.append(feature_vector, cell_feature_vector)
            #feature_vector.append(cell_feature_vector)
            #feature_vector = feature_vector.ravel()

    #print("Length of feature vector: " + str(len(feature_vector.tolist())))

    return feature_vector.tolist()


if __name__ == "__main__":

    #gradients = GradientDir()
    #gradients.compare_implemenatation()



    # gradients = GradientDir()
    # abs_gradient_x, abs_gradient_y, sign_flag = gradients._shuffle_signs(-255, -255)
    # bins_0, bins_1 = gradients._compute_dir(abs_gradient_x, abs_gradient_y, sign_flag)
    # gradient_dir = gradients._combine_dir(bins_0, bins_1)
    # print("gradient_dir for -255, -255: " + str(hex(gradient_dir)))

    # image = np.zeros((44, 100))
    image = np.zeros((64, 128))
    image[20:24, 40:50] = 255
    # print(image[19:25, 39:51])
    # gradients.compute(image)

    hog(image)
