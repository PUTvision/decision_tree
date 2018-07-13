from typing import Tuple
from sklearn import datasets
from sklearn.utils import shuffle
import scipy.io
import scipy.stats
import scipy.fftpack
import numpy as np

from decision_trees.datasets.dataset_base import DatasetBase


class Terrain(DatasetBase):
    def __init__(self, path: str):
        self.path = path

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mat = scipy.io.loadmat(f'{self.path}pociete_kroki_rasp.mat')

        # for key, value in mat.items():
        #     print(key)

        steps_raw = {}
        steps_raw

        krok_black = mat['krok_black']
        krok_deski = mat['krok_deski']
        krok_kam = mat['krok_kam']
        krok_pcv = mat['krok_pcv']
        krok_plytki = mat['krok_plytki']
        krok_wyk = mat['krok_wyk']

        print(np.shape(krok_black))
        print(np.shape(krok_deski))
        print(np.shape(krok_kam))
        print(np.shape(krok_pcv))
        print(np.shape(krok_plytki))
        print(np.shape(krok_wyk))

        def do_fft(signal):
            yf = scipy.fftpack.fft(signal, axis=0)
            yf = 2.0 / len(signal) * np.abs(yf[0:len(signal) // 2])
            return yf

        def compute_stats(matrix):
            vector = []
            for i in range(0, np.shape(matrix)[0]):
                data_for_processing = matrix[i, :401, :]
                variance = np.var(data_for_processing, axis=0)
                skew = scipy.stats.skew(data_for_processing, axis=0)
                kurtosis = scipy.stats.kurtosis(data_for_processing, axis=0)
                fifth_moment = scipy.stats.moment(data_for_processing, moment=5, axis=0)
                temp = np.array([variance, skew, kurtosis, fifth_moment])
                fouriers = do_fft(data_for_processing)
                temp = np.vstack((temp, fouriers[1:25, :]))
                if vector == []:
                    vector = temp
                else:
                    vector = np.dstack((vector, temp))
            vector = np.transpose(vector)
            return vector

        # 1st dim - step index
        # 2nd dim - measurement [F_x, F_y, F_z, T_x, T_y, T_z]
        # 3rd dim - [variance, skew, kurtosis, fifth_moment, 24 first fft elements sans zero frequency]

        sample = compute_stats(krok_deski)

        print(np.shape(sample))
        print(sample[0, :, :])
        print(sample[0, 0, :])

        steps_concatenetad = []
        for step in sample:
            print(np.shape(step))
            concatenated = np.concatenate(step)
            steps_concatenetad.append(concatenated)

        print(np.shape(steps_concatenetad))

    @staticmethod
    def _normalise(data: np.ndarray):
        # TODO: normalise each parameter (variance / skew / ...) for Fx, Fy etc separately
        # in case of MNIST data it is possible to just divide each data by maximum value
        # each feature is in range 0-255
        data = data / 255

        return data


if __name__ == "__main__":
    d = Terrain("./data/")
    d._load_data()
