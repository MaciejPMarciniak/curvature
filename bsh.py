import numpy as np
import pandas as pd
import glob
import os
import csv
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.interpolate import Rbf
from curvature import GradientCurvature
from LV_edgedetection import check_directory
from plotting import PlottingCurvature, PlottingDistributions


class Trace:

    def __init__(self, source_path, case_name, contours=None, interpolate=None, view='4C'):
        self.source_path = source_path
        self.case_name = case_name
        self.view = view
        self.id = None
        self.number_of_frames = 0
        self.number_of_points = 0
        self.es_frame = 0
        self.ed_frame = 0
        self.apex = 0
        self.ventricle_curvature = []
        self.mean_curvature_over_time = []
        self.apices = []

        if contours is not None:  # Contours could be provided if coming directly from images and LV_edgedetection
            self.data = contours
            self.case_filename = case_name
        else:  # Normally, they are read from EchoPAC export
            self.case_filename = self.case_name.split('/')[-1][:-4]
            self.data = self._read_echopac_output()

        if interpolate is not None:
            assert interpolate >= self.number_of_points, ('Number of points to interpolate is too low, must be at least'
                                                          'the length of the underlying contour')
            self._interpolate_traces(interpolate)
        self.get_curvature_per_frame()
        print('CURVATURE_CALCULATED')
        self.vc_normalized = self.get_normalized_curvature(self.ventricle_curvature)
        self._find_ed_and_es_frame()
        self.find_apices()
        self.biomarkers = pd.DataFrame(index=[self.id])
        self.get_biomarkers()
        print('BIOMARKERS OBTAINED')

    def _read_echopac_output(self):
        file_w_path = os.path.join(self.source_path, self.case_name)
        with open(file_w_path) as f:
            for line in f:
                if 'ID=' in line:
                    self.id = line.split('=')[1].strip('\n,')
                    self.id += '_' + '_'.join(self.case_filename.split('_')[-3:])
                    print(self.id)
                    f.close()
                    break

        data = pd.read_csv(filepath_or_buffer=file_w_path, sep=',', skiprows=10, header=None,
                           delim_whitespace=False)
        data.dropna(axis=1, inplace=True)
        data = data.values / 10  # mm to cm. Original values are given in millimeters.
        self.number_of_frames, self.number_of_points = data.shape[0], int(data.shape[1]/2)
        return data

    @staticmethod
    def _plane_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def _trace_length(x, y):
        set1 = np.vstack([x[:-1], y[:-1]]).T
        set2 = np.vstack([x[1:], y[1:]]).T
        return np.sum([np.linalg.norm(a - b) for a, b in zip(set1, set2)])

    @staticmethod
    def get_normalized_curvature(curvature):
        # Empirically established values. Useful for coloring, where values cannot be negative.
        return [(single_curvature + 1.5) / 3 for single_curvature in curvature]

    def _find_ed_and_es_frame(self):
        areas = np.zeros(int(self.number_of_frames/2))  # Search only in the first half for ED
        for f, frame in enumerate(range(int(self.number_of_frames/2))):
            x, y = self.data[frame, ::2], self.data[frame, 1::2]
            print(f)
            areas[f] = self._plane_area(x, y)
        self.es_frame, self.ed_frame = np.argmin(areas), np.argmax(areas)

    def _interpolate_traces(self, trace_points_n=None):

        if trace_points_n is None:  # Only change the format of the data
            point_interpolated = np.zeros((len(self.data), len(self.data[0]) * 2))
            for trace in range(self.data.shape[0]):  # number of frames
                point_interpolated[trace, ::2] = [x[0] for x in self.data[trace]]
                point_interpolated[trace, 1::2] = [y[1] for y in self.data[trace]]
        else:  # perform interpolation
            point_interpolated = np.zeros((len(self.data), trace_points_n * 2))
            for trace in range(self.data.shape[0]):  # number of frames
                positions = np.arange(int(self.data.shape[1]/2))  # strictly monotonic, number of points in single trace
                interpolation_target_n = np.linspace(0, self.data.shape[1]/2 - 1, trace_points_n)
                # Radial basis function interpolation 'quintic': r**5 where r is the distance from the next point
                # Smoothing is set to length of the input data
                rbf_x = Rbf(positions, self.data[trace, ::2], smooth=len(positions), function='quintic')
                rbf_y = Rbf(positions, self.data[trace, 1::2], smooth=len(positions), function='quintic')
                # Interpolate based on the RBF model
                point_interpolated[trace, ::2] = rbf_x(interpolation_target_n)
                point_interpolated[trace, 1::2] = rbf_y(interpolation_target_n)

                # a = x[0] for x in self.data[trace])
                # plt.plot(self.data[trace][::2], -self.data[trace][1::2], '.-')
                # plt.plot(point_interpolated[trace][::2], -point_interpolated[trace][1::2], 'r')
                # plt.show()
                # exit()

        self.data = point_interpolated
        self.number_of_frames, self.number_of_points = self.data.shape  # adjust to interpolated data shape

        return point_interpolated

    def get_curvature_per_frame(self):
        for frame in range(self.number_of_frames):  # Points should be interpolated already
            trace = np.array([[x, y] for x, y in zip(self.data[frame][::2], self.data[frame][1::2])])  # 2D data: x,y
            curvature = GradientCurvature(trace=trace)
            self.ventricle_curvature.append(curvature.calculate_curvature())
            # plt.figure()
            # plt.plot(self.ventricle_curvature[-1])
            # plt.show()
        self.ventricle_curvature = np.array(self.ventricle_curvature)

    def get_mean_curvature_over_time(self):
        self.mean_curvature_over_time = np.mean(self.ventricle_curvature, axis=0)
        return self.mean_curvature_over_time

    def find_apices(self):
        for frame in range(self.number_of_frames):
            # self.apices.append(np.argmax(self.ventricle_curvature[frame]))  # Maximum curvature in a frame
            self.apices.append(np.argmin(self.data[frame, 1::2]))  # Lowest point in given frame
        values, counts = np.unique(self.apices, return_counts=True)
        self.apex = values[np.argmax(counts)]  # Lowest point in all frames

    def get_biomarkers(self):

        curv = pd.DataFrame(data=self.ventricle_curvature)

        if self.view == '2C':
            curv_min_col = curv.min(axis=0).idxmin()
            lower_bound = int(0.04 * len(curv.columns))
            upper_bound = int(0.96 * len(curv.columns))
        else:
            _t = int(self.view == '4C')
            lower_bound = int(np.round((_t * 0.04 + (1 - _t) * 0.7) * len(curv.columns)))
            upper_bound = int(np.round((_t * 0.3 + (1 - _t) * 0.96) * len(curv.columns)))

            # id of the column (point number) with minimum value
            curv_min_col = curv.loc[:, lower_bound:upper_bound].min(axis=0).idxmin()

        # Minimum values within the entire cycle
        self.biomarkers['min'] = curv.loc[:, lower_bound:upper_bound].min().min()
        self.biomarkers['min_delta'] = np.abs(curv[curv_min_col].max() - self.biomarkers['min'])
        # Average values in the cycle
        self.biomarkers['avg_min_basal_curv'] = curv.loc[:, lower_bound:upper_bound].mean().min()
        self.biomarkers['avg_avg_basal_curv'] = curv.loc[:, lower_bound:upper_bound].mean().mean()
        # Minimum values at ED
        self.biomarkers['min_ED'] = curv.loc[self.ed_frame, lower_bound:upper_bound].min()
        self.biomarkers['min_delta_ED'] = self.biomarkers.min_ED - self.biomarkers['min']
        self.biomarkers['avg_basal_ED'] = curv.loc[self.ed_frame, lower_bound:upper_bound].mean()
        print(self.biomarkers.avg_basal_ED)
        self.biomarkers['trace_length_ED'] = self._trace_length(self.data[self.ed_frame, ::2],
                                                                self.data[self.ed_frame, 1::2])
        print(self.biomarkers)
        return self.biomarkers


class Cohort:

    def __init__(self, source_path='data', view='4C', output_path='data', indices_file='indices_all_cases.csv',
                 interpolate_traces=None):

        self.view = view
        self.source_path = source_path
        self.output_path = check_directory(output_path)
        self.indices_file = indices_file
        self.interpolate_traces = interpolate_traces
        self.files = glob.glob(os.path.join(self.source_path, '*.CSV'))
        self.files.sort()

        self.df_all_cases = None
        self.df_master = None
        self.curv = None
        self.biomarkers = None
        self.table_name = None

    def _set_paths_and_files(self, view=None, output_path=''):
        # self.view = view
        if view is not None:  # Read specified view data
            self.files = glob.glob(os.path.join(self.source_path, self.view, '*.CSV'))
        else:  # Read all CSVs in the source directory
            self.files = glob.glob(os.path.join(self.source_path, '*.CSV'))
        self.files.sort()
        if not output_path == '':
            self.output_path = check_directory(output_path)

    def _try_get_data(self, data=False, master_table=False):

        if data:
            _data_file = os.path.join(self.output_path, 'output_EDA', self.table_name)

            if os.path.isfile(_data_file):
                self.df_all_cases = pd.read_csv(_data_file, header=0, index_col=0)

            if not os.path.exists(_data_file):
                self._build_data_set(to_file=True)
            self.biomarkers = list(self.df_all_cases.columns)

        if master_table:

            _master_table_file = os.path.join(self.output_path, self.table_name)

            if os.path.isfile(_master_table_file):
                self.df_master = pd.read_csv(_master_table_file, header=0, index_col=0)

            if not os.path.exists(_master_table_file):
                self._build_master_table(to_file=True)

            if self.view is not None:
                self.biomarkers = list(set([col[3:] for col in self.df_master.columns]))

        if 'label' in self.biomarkers:
            self.biomarkers.remove('label')

        if not (data or master_table):
            exit('No data has been created, set the data or master_table parameter to True')

    def _build_data_set(self, to_file=False):

        list_of_dfs = []
        df_patient_data = pd.read_excel(os.path.join(patient_data_path, 'PREDICT-AF_Measurements.xlsx'),  # AduHeart_PatientData_Relevant
                                        index_col='patient_ID', header=0)

        for curv_file in self.files:
            print('case: {}'.format(curv_file))
            ven = Trace(self.source_path, case_name=curv_file, view=self.view, interpolate=self.interpolate_traces)
            list_of_dfs.append(ven.get_biomarkers())

        self.df_all_cases = pd.concat(list_of_dfs)

        self.df_all_cases.index.name = 'ID'
        self.df_all_cases['patient_ID'] = self.df_all_cases.index.map({k: k.split('_')[0] for k
                                                                       in self.df_all_cases.index})
        print(self.df_all_cases)
        print(df_patient_data)
        self.df_all_cases = self.df_all_cases.join(df_patient_data.SB, how='inner', on='patient_ID')
        self.df_all_cases = self.df_all_cases.set_index(['patient_ID', self.df_all_cases.index])

        self.df_all_cases['min_index'] = np.abs(self.df_all_cases.min_delta * self.df_all_cases['min'])
        self.df_all_cases['min_index_ED'] = np.abs(self.df_all_cases.min_delta_ED * self.df_all_cases.min_ED)
        self.df_all_cases['curv_len_inter'] = np.abs(self.df_all_cases.min_ED * self.df_all_cases.trace_length_ED)

        print(self.df_all_cases)
        if to_file:
            data_set_output_dir = check_directory(os.path.join(self.output_path, 'output_EDA'))
            self.df_all_cases.to_csv(os.path.join(data_set_output_dir, self.indices_file))
            print('master table saved')

    def _build_master_table(self, to_file=False, views=(None,)):

        list_of_dfs = []

        for view in views:
            self._set_paths_and_files(view=view, output_path=self.output_path)
            self._try_get_data(data=True)
            # self.df_all_cases.columns = [(self.view if self.view is not None else '')
            #                              + '_' + col for col in self.df_all_cases.columns]
            list_of_dfs.append(self.df_all_cases)

        self.df_master = list_of_dfs[0]
        for df_i in range(1, len(list_of_dfs)):
            self.df_master = self.df_master.merge(list_of_dfs[df_i], right_index=True, left_index=True, sort=True)
        self.df_master.index.names = [self.df_master.index.names[0], 'patient_ID_detail']
        self.df_master.to_csv(os.path.join(self.output_path, 'master_table_full.csv'))

        if to_file:
            df_min_ed = self.df_master.groupby(level=0).min_ED.min()
            df_min_ed.to_csv(os.path.join(self.output_path, 'master_table.csv'))

    def _plot_master(self):

        if self.df_master is None:
            self._try_get_data(master_table=True)

        _master_output_path = check_directory(os.path.join(self.output_path, 'output_master'))
        master_plot_tool = PlottingDistributions(self.df_master, '', _master_output_path)

        for col in self.biomarkers:
            print(col)
            if self.table_name != 'master_table.csv':
                master_plot_tool.plot_with_labels('4C_' + col, '3C_' + col)
            else:
                master_plot_tool.plot_2_distributions('4C_' + col, '3C_' + col, kind='kde')

    def _plot_data(self):

        if self.df_all_cases is None:
            self._try_get_data(data=True)

        _view_output_path = check_directory(os.path.join(self.output_path, self.view, 'output_EDA'))

        plot_tool = PlottingDistributions(self.df_all_cases, '', _view_output_path)
        for col in self.biomarkers:
            plot_tool.set_series(col)
            plot_tool.plot_distribution()

        col_combs = combinations(self.biomarkers, 2)
        for comb in col_combs:
            if self.table_name != 'master_table.csv':
                plot_tool.plot_with_labels(comb[0], comb[1])
            else:
                plot_tool.plot_2_distributions(comb[0], comb[1], kind='kde')

    def print_names_and_ids(self, to_file=False, views=('4C', '3C', '2C')):

        for view in views:
            self._set_paths_and_files(view=view)
            names = {}
            for curv_file in self.files:
                print('case: {}'.format(curv_file))
                ven = Trace(self.source_path, case_name=curv_file, view=self.view, interpolate=self.interpolate_traces)
                case_name = curv_file.split('/')[-1].split('.')[0]  # get case name without path and extension
                names[case_name] = ven.id
            if to_file:
                with open(os.path.join(self.output_path,
                                       'Names_IDs_' + (self.view if self.view is not None else '') + '.csv'), 'w') as f:
                    w = csv.DictWriter(f, names.keys())
                    w.writeheader()
                    w.writerow(names)

    def save_curvatures(self):

        _output_path = check_directory(os.path.join(self.output_path, 'curvatures'))

        for case in self.files:
            ven = Trace(self.source_path, case_name=case, view=self.view, interpolate=self.interpolate_traces)
            print(ven.id)
            pd.DataFrame(ven.ventricle_curvature).to_csv(os.path.join(_output_path, ven.id+'.csv'))

    def save_indices(self):

        self._build_data_set(to_file=True)

    def save_statistics(self):

        if self.df_master is None:
            self.table_name = 'master_table_with_labels.csv'
            self._try_get_data(master_table=True)

        print(self.df_master)
        df_stats = pd.DataFrame()
        for lab in range(3):
            df_stats['mean_'+str(lab)] = self.df_master[self.df_master['SB'] == lab].mean()
        for lab in range(3):
            df_stats['std_' + str(lab)] = self.df_master[self.df_master['SB'] == lab].std()
        df_stats.to_csv(os.path.join(self.output_path, 'master_stats.csv'))

    def save_extemes(self, n=30):

        self._try_get_data(data=True)

        list_of_extremes = []
        for col in self.df_all_cases.columns:
            list_of_extremes.append(self.df_all_cases[col].sort_values(ascending=False).index.values[:n])
            list_of_extremes.append(self.df_all_cases[col].sort_values(ascending=False).values[:n])

        index_lists = [2 * [i] for i in self.df_all_cases.columns]
        index = [item for sublist in index_lists for item in sublist]

        df_extremes = pd.DataFrame(list_of_extremes, index=index)
        _output_path = check_directory(os.path.join(self.output_path, self.view, 'output_EDA'))
        df_extremes.to_csv(os.path.join(_output_path, 'extremes.csv'))

    def plot_curvatures(self, coloring_scheme='curvature', plot_mean=False):

        _source_path = os.path.join(self.source_path, self.view)
        if plot_mean:
            _output_path = check_directory(os.path.join(self.output_path, 'output_curvature', 'mean'))
        else:
            _output_path = check_directory(os.path.join(self.output_path, 'output_curvature'))

        for case in self.files:
            ven = Trace(self.source_path, case_name=case, view=self.view, interpolate=self.interpolate_traces)
            print(ven.id)
            print('Points: {}'.format(ven.number_of_points))
            plot_tool = PlottingCurvature(source=_source_path,
                                          output_path=_output_path,
                                          ventricle=ven)
            if plot_mean:
                plot_tool.plot_mean_curvature()
            else:
                plot_tool.plot_all_frames(coloring_scheme=coloring_scheme)
                plot_tool.plot_heatmap()

    def plot_distributions(self, plot_data=False, plot_master=False, table_name=None):

        self.table_name = table_name

        if plot_data:
            self._plot_data()

        if plot_master:
            self._plot_master()


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Cohort(windows)
    patient_data_path = os.path.join('C:/', 'Data', 'ProjectCurvature', 'PatientData')
    source = os.path.join('C:/', 'Data', 'ProjectCurvature', 'InterObserverStudy')
    target = os.path.join('C:/', 'Data', 'ProjectCurvature', 'InterObserverStudy', 'Output')

    # cohort = Cohort(source_path=source, view='4C', output_path=target, interpolate_traces=500)
    plot_path = r'C:\Data\ProjectCurvature\Analysis\EndoContours'
    case = '2DS120_RRC0115_RODRIGUEZ RIOS_26_05_2017_4CH_FULL_TRACE_ENDO_V1_D2_B.CSV'
    ven = Trace(plot_path, case, interpolate=500)
    plot_tool = PlottingCurvature(source='.',
                                  output_path='C:\Code\curvature\images',
                                  ventricle=ven)
    plot_tool.plot_all_frames(coloring_scheme='curvature')
    # plot_tool.plot_heatmap()

    # cohort.save_curvatures()
    # cohort.save_indices()
    # cohort.plot_curvatures()
    # cohort.print_names_and_ids(to_file=True, views=(None,))
    # cohort.save_statistics()

    # for i, f in enumerate(os.listdir(source)):
    #
    #     case = f.split('_')[1]
    #     print(case)
    #     curv = Trace(source_path=source, case_name=f, interpolate=500)

    # Cohort(linux)
    # source = os.path.join('home','mat','Python','data','curvature')
    # target_path = os.path.join('home','mat','Python','data','curvature')

    # representatives = ('RV_4C.CSV', 'CAS0214_4C.CSV', 'DPJMA0472.CSV')
    #
    # for _view in ['4C']:
    #
    #     cohort = Cohort(source_path=source, view=_view, output_path=target_path)
    #
    # cohort.plot_curvatures('asf')
    # cohort.save_curvatures()
    # cohort.plot_curvatures(coloring_scheme='curvature', plot_mean=False)
    # cohort.plot_distributions(plot_data=True, table_name='_all_cases_with_labels.csv')
    # cohort.print_names_and_ids(to_file=True)
    #
    # _view = '4C'
    # case_name = os.path.join(source, _view, 'AFI0442_4C.CSV')
    # ven = Trace(case_name, view=_view)
    # plot_tool = PlottingCurvature(source=source, output_path=target_path, ventricle=ven)
    # plot_tool.plot_mean_curvature()
    #
    # print(ven.number_of_points)
    # print(ven.number_of_frames)
    # ------------------------------------------------------------------------------------------------------------------
