import numpy as np


def convert_to_fixed_point(float_value: float, n_bits: int) -> float:
    f = (1 << n_bits)

    return np.round(float_value * f) * (1.0 / f)


def test_convert_to_fixed_point():
    assert convert_to_fixed_point(1 / 3, 2) == 0.25
    assert convert_to_fixed_point(1 / 3, 3) == 0.375


if __name__ == "__main__":
    value = 13 / 16
    print(value)

    for i in range(10, 0, -1):
        value_as_fixed_point = convert_to_fixed_point(value, i)
        print("Number of bits used: " + str(i) + " value: " + str(value_as_fixed_point))
        print("% error: {0:.2f}".format(abs(value - value_as_fixed_point) / value * 100))
