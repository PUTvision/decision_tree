from typing import Tuple
from sklearn.model_selection import train_test_split
import scipy.io
import scipy.stats
import scipy.fftpack
import numpy as np

from decision_trees.datasets.dataset_base import DatasetBase
from decision_trees.gridsearch import perform_gridsearch
from decision_trees.utils.constants import ClassifierType, GridSearchType


class Terrain(DatasetBase):
    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mat = scipy.io.loadmat(f'{self.path}pociete_kroki_rasp.mat')

        # for key, value in mat.items():
        #     print(key)

        steps_raw = {
            'krok_black': mat['krok_black'],
            'krok_deski': mat['krok_deski'],
            'krok_kam': mat['krok_kam'],
            'krok_pcv': mat['krok_pcv'],
            'krok_plytki': mat['krok_plytki'],
            'krok_wyk': mat['krok_wyk']
        }

        # for key, value in steps_raw.items():
        #     print(f'{key} shape: {np.shape(value)}')

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

        input_data = []
        output_data = []

        for idx, (key, value) in enumerate(steps_raw.items()):
            # print(idx)
            steps_stats = compute_stats(value)

            # print(np.shape(step_stats))
            # print(step_stats[0, :, :])
            # print(step_stats[0, 0, :])

            steps_stats_concatenated = []
            for s in steps_stats:
                # print(np.shape(s))
                s_concatenated = np.concatenate(s)
                # print(np.shape(s_concatenated))
                steps_stats_concatenated.append(s_concatenated)

            # print(np.shape(steps_stats_concatenated))
            input_data.extend(steps_stats_concatenated)
            output_data.extend(np.full(np.shape(steps_stats_concatenated)[0], idx))

        print(np.shape(input_data))
        print(np.shape(output_data))

        input_data = self._normalise(np.asarray(input_data))

        train_data, test_data, train_target, test_target = train_test_split(
            input_data, output_data, test_size=0.2, random_state=42, shuffle=True
        )

        return np.asarray(train_data), np.asarray(train_target), np.asarray(test_data), np.asarray(test_target)

    @staticmethod
    def _normalise(data: np.ndarray):
        # normalise each parameter (variance / skew / ...) for Fx, Fy etc separately

        # col_idx = 167
        #
        # print(f'max: {np.max(data[:, col_idx])}')
        # print(f'min: {np.min(data[:, col_idx])}')
        # print(np.shape(data[:, col_idx]))

        data_normed = (data - np.min(data, 0)) / np.ptp(data, 0)

        # print(f'max: {np.max(data_normed[:, col_idx])}')
        # print(f'min: {np.min(data_normed[:, col_idx])}')
        # print(np.shape(data_normed[:, col_idx]))

        return data_normed


def main():
    d = Terrain('./../../data/datasets/terrain_data/')
    train_data, train_target, test_data, test_target = d.load_data()
    print(f'np.shape(train_data): {np.shape(train_data)}')
    print(f'np.unique(test_target): {np.unique(test_target)}')

    d.test_as_classifier(2, './../../data/vhdl/')

    # perform_gridsearch(train_data, train_target, test_data, test_target,
    #                    [16, 12, 8, 6, 4, 2, 1],
    #                    ClassifierType.RANDOM_FOREST,
    #                    GridSearchType.NONE,
    #                    './../../data/gridsearch_results/',
    #                    d.__class__.__name__
    #                    )


if __name__ == '__main__':
    main()
