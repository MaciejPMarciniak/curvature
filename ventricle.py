import numpy as np
import pandas as pd
import glob
import os
from curvature import Curvature
from plotting import Plotting
import matplotlib.pyplot as plt


class Ventricle:

    def __init__(self, case_name):
        self.case_name = case_name
        self.id = None
        self.number_of_frames = 0
        self.number_of_points = 0
        self.es_frame = 0
        self.ed_frame = 0
        self.apex = 0
        self.data = self.read_echopac_output()
        self.ventricle_curvature = []
        self.vc_normalized = []  # ventricle_curvature normalized
        self.apices = []
        self.get_curvature_per_frame()
        self.get_normalized_curvature()
        self.find_apices()
        self.find_ed_and_es_frame()

    def read_echopac_output(self):
        with open(self.case_name) as f:
            for line in f:
                if 'ID=' in line:
                    self.id = line.split('=')[1].strip('\n')
                    f.close()
                    break

        data = pd.read_csv(filepath_or_buffer=self.case_name, sep=',', skiprows=10, header=None, delim_whitespace=False)
        data.dropna(axis=1, inplace=True)
        data = data.values
        self.number_of_frames, self.number_of_points = data.shape[0], int(data.shape[1]/2)
        return data

    @staticmethod
    def plane_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def find_ed_and_es_frame(self):
        areas = np.zeros(self.number_of_frames)
        for frame in range(self.number_of_frames):
            x, y = self.data[frame, ::2], self.data[frame, 1::2]
            areas[frame] = self.plane_area(x, y)
        self.es_frame, self.ed_frame = np.argmin(areas), np.argmax(areas)

    def get_curvature_per_frame(self):
        for frame in range(self.number_of_frames):
            xy = [[x, y] for x, y in zip(self.data[frame][::2], self.data[frame][1::2])]
            curva = Curvature(line=xy)
            self.ventricle_curvature.append(curva.calculate_curvature(gap=0))
        self.ventricle_curvature = np.array(self.ventricle_curvature)

    def find_apices(self):
        for frame in range(self.number_of_frames):
            # self.apices.append(np.argmax(self.ventricle_curvature[frame]))  # Maximum curvature in given frame
            self.apices.append(np.argmin(self.data[frame, 1::2]))  # Lowest point in given frame
        values, counts = np.unique(self.apices, return_counts=True)
        self.apex = values[np.argmax(counts)]  # Lowest point in all frames
        # self.apex = self.apices[self.ed_frame]  # Lowest point at end diastole
        print(self.apex)
        print(self.number_of_points)

    def get_normalized_curvature(self):
        # Empirically established values
        self.vc_normalized = [(single_curvature+0.125)*4 for single_curvature in self.ventricle_curvature]

    def move_to_apex(self):
        pass


if __name__ == '__main__':

    source = '/home/mat/Python/data/echo_delineation'
    cases = glob.glob(os.path.join(source, '*.CSV'))
    cases.sort()

    max_curves = []
    min_curves = []
    for case in cases:
        ven = Ventricle(case_name=case)
        print(ven.id)
        # print('Points: {}'.format(ven.number_of_points))
        plotting = Plotting(source=source, ventricle=ven)
        plotting.plot_all_frames(coloring_scheme='curvture')

    # plt.plot(max_curves)
    # plt.plot(min_curves)
    # print(min(min_curves), min_curves.index(min(min_curves)))
    # plt.show()
