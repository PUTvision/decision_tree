__author__ = 'Amin'

import numpy as np


class MF_lbp:

    def __init__(self, use_test_version=False):

        if use_test_version:
            self.encoded_lbp_lut = [0] * 256
            self.encoded_lbp_lut[16] = 11
        else:
            self.encoded_lbp_lut = [29] * 256
            self.encoded_lbp_lut[0], self.encoded_lbp_lut[255] = 0, 0
            self.encoded_lbp_lut[1], self.encoded_lbp_lut[254] = 1, 1
            self.encoded_lbp_lut[2], self.encoded_lbp_lut[253] = 2, 2
            self.encoded_lbp_lut[3], self.encoded_lbp_lut[252] = 3, 3
            self.encoded_lbp_lut[4], self.encoded_lbp_lut[251] = 4, 4
            self.encoded_lbp_lut[6], self.encoded_lbp_lut[249] = 5, 5
            self.encoded_lbp_lut[7], self.encoded_lbp_lut[248] = 6, 6
            self.encoded_lbp_lut[8], self.encoded_lbp_lut[247] = 7, 7
            self.encoded_lbp_lut[12], self.encoded_lbp_lut[243] = 8, 8
            self.encoded_lbp_lut[14], self.encoded_lbp_lut[241] = 9, 9
            self.encoded_lbp_lut[15], self.encoded_lbp_lut[240] = 10, 10
            self.encoded_lbp_lut[16], self.encoded_lbp_lut[239] = 11, 11
            self.encoded_lbp_lut[24], self.encoded_lbp_lut[231] = 12, 12
            self.encoded_lbp_lut[28], self.encoded_lbp_lut[227] = 13, 13
            self.encoded_lbp_lut[30], self.encoded_lbp_lut[225] = 14, 14
            self.encoded_lbp_lut[31], self.encoded_lbp_lut[224] = 15, 15
            self.encoded_lbp_lut[32], self.encoded_lbp_lut[223] = 16, 16
            self.encoded_lbp_lut[48], self.encoded_lbp_lut[207] = 17, 17
            self.encoded_lbp_lut[56], self.encoded_lbp_lut[199] = 18, 18
            self.encoded_lbp_lut[60], self.encoded_lbp_lut[195] = 19, 19
            self.encoded_lbp_lut[62], self.encoded_lbp_lut[193] = 20, 20
            self.encoded_lbp_lut[63], self.encoded_lbp_lut[192] = 21, 21
            self.encoded_lbp_lut[64], self.encoded_lbp_lut[191] = 22, 22
            self.encoded_lbp_lut[96], self.encoded_lbp_lut[159] = 23, 23
            self.encoded_lbp_lut[112], self.encoded_lbp_lut[243] = 24, 24
            self.encoded_lbp_lut[120], self.encoded_lbp_lut[135] = 25, 25
            self.encoded_lbp_lut[124], self.encoded_lbp_lut[131] = 26, 26
            self.encoded_lbp_lut[126], self.encoded_lbp_lut[129] = 27, 27
            self.encoded_lbp_lut[127], self.encoded_lbp_lut[128] = 28, 28

    def calc_nrulbp_3x3(self, image):
    
        output_shape = (image.shape[0], image.shape[1])
        nrulbp_3x3_image = np.zeros(output_shape, dtype=np.double)
    
        rows = image.shape[0]
        cols = image.shape[1]
    
        for r in range(1, rows-1):
            for c in range(1, cols-1):
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
    
                nrulbp_3x3_image[r, c] = self.encoded_lbp_lut[raw_lbp_descriptor]
    
        return np.asarray(nrulbp_3x3_image)
