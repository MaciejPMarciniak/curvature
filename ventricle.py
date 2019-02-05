import numpy as np
import pandas as pd
import glob
import os
from curvature import Curvature
from plotting import PlottingCurvature, PlottingDistributions
import matplotlib.pyplot as plt
from itertools import combinations


class Ventricle:

    def __init__(self, case_name):
        self.case_name = case_name
        self.id = None
        self.number_of_frames = 0
        self.number_of_points = 0
        self.es_frame = 0
        self.ed_frame = 0
        self.apex = 0
        self.ventricle_curvature = []
        self.vc_normalized = []  # ventricle_curvature normalized
        self.apices = []
        self.data = self._read_echopac_output()
        self.biomarkers = pd.DataFrame(index=[self.id])
        self.get_curvature_per_frame()
        self.get_normalized_curvature()
        self.find_apices()
        self.find_ed_and_es_frame()

    def _read_echopac_output(self):
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
    def _plane_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def find_ed_and_es_frame(self):
        areas = np.zeros(self.number_of_frames)
        for frame in range(self.number_of_frames):
            x, y = self.data[frame, ::2], self.data[frame, 1::2]
            areas[frame] = self._plane_area(x, y)
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
        # print(self.apex)
        # print(self.number_of_points)

    def get_normalized_curvature(self):
        # Empirically established values. Useful for coloring, where values cannot be negative.
        self.vc_normalized = [(single_curvature+0.125)*4 for single_curvature in self.ventricle_curvature]

    def get_biomarkers(self):

        curv = pd.DataFrame(data=self.ventricle_curvature)
        curv_min_col = curv.min(axis=0).idxmin()
        curv_max_col = curv.max(axis=0).idxmax()
        curv_min_row = curv.min(axis=1).idxmin()

        self.biomarkers['min'] = curv.min().min()
        self.biomarkers['max'] = curv.max().max()
        self.biomarkers['min_delta'] = np.abs(curv[curv_min_col].max() - self.biomarkers['min'])
        self.biomarkers['max_delta'] = np.abs(curv[curv_max_col].min() - self.biomarkers['max'])
        self.biomarkers['min_index'] = np.abs(self.biomarkers['min_delta'] / self.biomarkers['min'])
        self.biomarkers['max_index'] = np.abs(self.biomarkers['max_delta'] / self.biomarkers['max'])
        self.biomarkers['amplitude_at_t'] = np.abs(curv.loc[curv_min_row].max() - self.biomarkers['min'])
        self.biomarkers['min_v_amp_index'] = self.biomarkers['min_delta'] / self.biomarkers['amplitude_at_t']
        self.biomarkers['delta_ratio'] = self.biomarkers['min_delta'] / self.biomarkers['max_delta']

        return self.biomarkers


class Cohort:

    def __init__(self, source_path='data', view='4C', output_path='data', output='all_cases.csv'):
        self.source_path = source_path
        self.view = view
        self.output_path = output_path
        self.output = output
        self.files = glob.glob(os.path.join(self.source_path, '*.CSV'))
        self.files.sort()
        self.df_all_cases = None
        self.curv = None
        self._check_directory(self.output_path)
        self._check_directory(os.path.join(self.output_path, self.view))

    @staticmethod
    def _check_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    def _try_get_data(self):

        data_file = os.path.join(self.output_path, self.view, 'output_EDA', self.output)
        if not os.path.exists(data_file):
            self._build_data_set(to_file=True)

        if self.df_all_cases is None:
            self.df_all_cases = pd.read_csv(data_file, header=0, index_col=0)

    def _build_data_set(self, to_file=False):

        list_of_dfs = []
        for curv_file in self.files:
            print('case: {}'.format(curv_file))
            ven = Ventricle(curv_file)
            list_of_dfs.append(ven.get_biomarkers())

        self.df_all_cases = pd.concat(list_of_dfs)
        if to_file:
            self.df_all_cases.to_csv(os.path.join(self.output_path, self.view, self.output))

    def plot_curvatures(self):

        c_scheme = 'curvature'
        self._check_directory(os.path.join(self.output_path, self.view, 'output_' + c_scheme))

        for case in self.files:
            ven = Ventricle(case_name=case)
            print(ven.id)
            print('Points: {}'.format(ven.number_of_points))
            plot_tool = PlottingCurvature(source=self.source_path,
                                          output_path=os.path.join(self.output_path, self.view, 'output_' + c_scheme),
                                          ventricle=ven)
            plot_tool.plot_all_frames(coloring_scheme=c_scheme)

    def plot_distributions(self):

        if self.df_all_cases is None:
            self._try_get_data()
        self._check_directory(os.path.join(self.output_path, self.view, 'output_EDA'))

        plot_tool = PlottingDistributions(self.df_all_cases, '', os.path.join(self.output_path, self.view, 'output_EDA'))
        for col in self.df_all_cases.columns:
            plot_tool.set_series(col)
            plot_tool.plot_distribution()

        col_combs = combinations(self.df_all_cases.columns, 2)
        for comb in col_combs:
            plot_tool.plot_2_distributions(comb[0], comb[1], kind='kde')


if __name__ == '__main__':

    view = '2C'
    source = os.path.join('/home/mat/Python/data/curvature', view)
    target_path = os.path.join('/home/mat/Python/data/curvature/')

    cohort = Cohort(source_path=source, view=view, output_path=target_path)
    # cohort.plot_curvatures()
    cohort.plot_distributions()
