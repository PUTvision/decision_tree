__author__ = 'Amin'

import numpy as np


class GradientDir:

    def __init__(self):
        pass

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
                #print("pixel_descriptor: " + str(pixel_descriptor))

                pixel_descriptors[row][col] = np.asarray(pixel_descriptor)

        #print("pixel_descriptors: \n" + str(pixel_descriptors))

        descriptors = self._gradient_adder(pixel_descriptors)
        print("descriptors: \n" + str(descriptors))
        print("descriptors size: \n" + str(descriptors.shape))

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

    def _combine_dir(self, bins_0, bins_1):
        bin_number = 0b000000000000000000  # bit length: 18 (elements are indexed 0 through 17)
        for i in range(8, -1, -1):
            bins_0_bit = bins_0 & (1 << i)
            bins_1_bit = bins_1 & (1 << i)
            v1 = ((bins_0_bit & bins_1_bit) >> i)
            v2 = ((bins_0_bit ^ bins_1_bit) >> i)
            bin_number |= (v1 << (2*i+1))
            bin_number |= (v2 << (2*i))

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

    def _gradient_demux(self, gradient_dir, gradient_norm):
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

    def _gradient_adder(self, grad_in: np.ndarray):

        final_descriptor = np.zeros_like(grad_in)
        #print("grad_in.shape: " + str(grad_in.shape))

        row_start = 3
        row_end = 8 - (row_start + 1)

        col_start = 3
        col_end = 8 - (col_start + 1)

        print("Row: " + str(row_start) + ", " + str(row_end) + "\n")
        print("Col: " + str(col_start) + ", " + str(col_end) + "\n")

        for row in range(row_start, grad_in.shape[0]-row_end):
            for col in range(col_start, grad_in.shape[1]-col_end):

                subarray = np.asarray(grad_in[row-4:row+3+1, col-4:col+3+1])
                #print("subarray.shape: " + str(subarray.shape))

                sum = subarray.sum()
                #print("sum: " + str(sum))

                final_descriptor[row, col] = sum

        # each cell has 9 values, this function should compute the sum of all corresponding values
        return final_descriptor


gradients = GradientDir()

abs_gradient_x, abs_gradient_y, sign_flag = gradients._shuffle_signs(-255, -255)
bins_0, bins_1 = gradients._compute_dir(abs_gradient_x, abs_gradient_y, sign_flag)
gradient_dir = gradients._combine_dir(bins_0, bins_1)
print("gradient_dir for -255, -255: " + str(hex(gradient_dir)))

#image = np.arange(100).reshape(10, 10)
image = np.zeros((44, 100))
image[20:24, 40:50] = 255
#print(image[19:25, 39:51])
#print(image)
gradients.compute(image)

