import numpy as np


def convert_fixed_point_to_integer(fixed_point_float: float, n_bits: int) -> int:
    return int(round(fixed_point_float * 2**n_bits))


def convert_to_fixed_point(float_value: float, n_bits: int) -> float:
    f = (1 << n_bits)

    return np.round(float_value * f) * (1.0 / f)


def quantize_data(train_data: np.ndarray, test_data: np.ndarray, number_of_bits: int,
                  flag_save_details_to_file: bool = False, path: str = './'):
    train_data_quantized = np.array([convert_to_fixed_point(x, number_of_bits) for x in train_data])
    test_data_quantized = np.array([convert_to_fixed_point(x, number_of_bits) for x in test_data])

    if flag_save_details_to_file:
        with open(path + 'quantization_comparision.txt', 'w') as file_quantization:
            print(f'Train data size before quantization: {len(train_data)}', file=file_quantization)
            print(f'First element before quantization:\n' +
                  np.array2string(train_data[0], formatter={'float': lambda x: '%.3f' % x}),
                  file=file_quantization)
            print(f'Size after quantization: {len(train_data_quantized)}', file=file_quantization)
            print(f'First element after quantization:\n' +
                  np.array2string(train_data_quantized[0], formatter={'float': lambda x: '%.3f' % x}),
                  file=file_quantization)

    return train_data_quantized, test_data_quantized


def test_convert_to_fixed_point():
    assert convert_to_fixed_point(1 / 3, 2) == 0.25
    assert convert_to_fixed_point(1 / 3, 3) == 0.375


if __name__ == "__main__":
    value = 13 / 16
    print(value)

    for i in range(10, -1, -1):
        value_as_fixed_point = convert_to_fixed_point(value, i)
        print("Number of bits used: " + str(i) + " value: " + str(value_as_fixed_point))
        print("% error: {0:.2f}".format(abs(value - value_as_fixed_point) / value * 100))
