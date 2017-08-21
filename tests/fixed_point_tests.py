import numpy as np


def convert_to_fixed_point(float_value, n_bits):
    f = (1 << n_bits)

    return np.round(float_value * f) * (1.0 / f)

if __name__ == "__main__":
    value = 0.25984365873645
    print(value)

    for i in range(10, 1, -1):
        print("Number of bits used: " + str(i) + " value: " + str(convert_to_fixed_point(value, i)))
