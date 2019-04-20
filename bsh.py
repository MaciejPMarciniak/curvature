import numpy as np
import pandas as pd
import glob
import os
import csv
import pickle
import cv2
import shutil
from itertools import combinations
from scipy.interpolate import interp1d, Rbf
from pykalman import KalmanFilter
from openvino.inference_engine import IENetwork, IEPlugin
from PIL import Image
from curvature import Curvature
from LV_edgedetection import Contour
from plotting import PlottingCurvature, PlottingDistributions
import matplotlib.pyplot as plt


def check_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


class Trace:

    def __init__(self, case_name, contours, interpolation_parameters=(), view='4C'):
        self.case_name = case_name
        self.case_filename = self.case_name.split('/')[-1][:-4]
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
        self.biomarkers = pd.DataFrame(index=[self.case_name])

        if contours is not None:
            self.data = contours
        else:
            self.data = self._read_echopac_output()
        self._interpolate_traces(trace_points_n=interpolation_parameters[0],
                                 time_steps_n=interpolation_parameters[1])
        self.get_curvature_per_frame()
        print('CURVATURE_CALCULATED')
        self.vc_normalized = self.get_normalized_curvature(self.ventricle_curvature)
        self._find_ed_and_es_frame()
        self.find_apices()
        self.get_biomarkers()
        print('BIOMARKERS OBTAINED')

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
        # TODO: replace with an alborithm which takes into account the fact that some LVs are not convex
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _find_ed_and_es_frame(self):
        areas = np.zeros(self.number_of_frames)
        for frame in range(self.number_of_frames):
            x, y = self.data[frame, ::2], self.data[frame, 1::2]
            areas[frame] = self._plane_area(x, y)
        self.es_frame, self.ed_frame = np.argmin(areas), np.argmax(areas)

    def _interpolate_traces(self, trace_points_n=None, time_steps_n=None):

        # TODO: make a similar function that reads 2Dstrain exports (np array instead of list of tuple, the
        # TODO: same number of points in each frame

        if trace_points_n is None:
            point_interpolated = self.data

        else:
            print('len data: {}'.format(len(self.data)))
            x_time_steps = np.arange(len(self.data))
            point_interpolated = np.zeros((len(self.data), trace_points_n*2))

            for trace in x_time_steps:
                positions = np.arange(len(self.data[trace]))
                points_target = np.linspace(0, len(self.data[trace]) - 1, trace_points_n)
                # Cubic interpolation:
                # ci_x = interp1d(x=positions, y=self.data[trace, ::2], kind=7)
                # ci_y = interp1d(x=positions, y=self.data[trace, 1::2], kind=7)
                #
                # point_interpolated[trace, ::2] = ci_x(points_target)
                # point_interpolated[trace, 1::2] = ci_y(points_target)
                # -----
                # Spline interpolation (varying smoothness, depnedant on the number of points in original trace
                # (parameter s).
                # points_target = np.linspace(0, 1, trace_points_n)
                # tck, u = splprep([self.data[trace, ::2], self.data[trace, 1::2]])
                # interpolation = splev(points_target, tck)
                # point_interpolated[trace, ::2] = interpolation[0]
                # point_interpolated[trace, 1::2] = interpolation[1]
                # -----
                # Radial basis function interpolation
                # 'quintic': r**5 where r is the distance from the next point
                # smoothing should be adjusted to the number of original points, but there is no clear
                # criterion
                rbf_x = Rbf(positions, [x[0] for x in self.data[trace]], smooth=40, function='quintic')
                rbf_y = Rbf(positions, [y[1] for y in self.data[trace]], smooth=40, function='quintic')

                point_interpolated[trace, ::2] = rbf_x(points_target)
                point_interpolated[trace, 1::2] = rbf_y(points_target)

            point_interpolated_array = np.array(point_interpolated)

            if time_steps_n is not None:
                time_steps_target = np.linspace(0, self.data.shape[0] - 1, time_steps_n)
                time_and_point_interpolated = np.zeros((time_steps_n, trace_points_n*2))

                for point in range(trace_points_n*2):
                    ci = interp1d(x=x_time_steps, y=point_interpolated[:, point], kind='cubic')
                    time_interp = ci(time_steps_target)
                    time_and_point_interpolated[:, point] = time_interp

                point_interpolated = time_and_point_interpolated

        print('Original resolution: {}, new resolution: {}'.format(len(self.data), point_interpolated_array.shape))

        self.number_of_frames, self.number_of_points = point_interpolated.shape
        self.data = point_interpolated

    def get_curvature_per_frame(self):
        for frame in range(self.number_of_frames):
            xy = [[x, y] for x, y in zip(self.data[frame][::2], self.data[frame][1::2])]
            curva = Curvature(line=xy)
            self.ventricle_curvature.append(curva.calculate_curvature(gap=0))

        self.ventricle_curvature = np.array(self.ventricle_curvature)

    def get_mean_curvature_over_time(self):
        self.mean_curvature_over_time = np.mean(self.ventricle_curvature, axis=0)
        return self.mean_curvature_over_time

    def find_apices(self):
        for frame in range(self.number_of_frames):
            # self.apices.append(np.argmax(self.ventricle_curvature[frame]))  # Maximum curvature in given frame
            self.apices.append(np.argmax(self.data[frame, 1::2]))  # Lowest point in given frame
        values, counts = np.unique(self.apices, return_counts=True)
        self.apex = values[np.argmax(counts)]  # Lowest point in all frames

    def get_normalized_curvature(self, curvature):
        # Empirically established values. Useful for coloring, where values cannot be negative.
        return [(single_curvature+0.125)*4 for single_curvature in curvature]

    def get_biomarkers(self):

        curv = pd.DataFrame(data=self.ventricle_curvature)

        if self.view == '2C':
            curv_min_col = curv.min(axis=0).idxmin()
            curv_min_row = curv.min(axis=1).idxmin()
            lower_bound = 0
            upper_bound = len(curv.columns)
        else:
            _t = int(self.view == '4C')
            lower_bound = int(np.round((_t * 0.04 + (1 - _t) * 0.7) * len(curv.columns)))
            upper_bound = int(np.round((_t * 0.3 + (1 - _t) * 0.96) * len(curv.columns)))
            print('lower bound: {}, upper bound: {}'.format(lower_bound, upper_bound))

            # id of the column (point number) with minimum value
            curv_min_col = curv.loc[:, lower_bound:upper_bound].min(axis=0).idxmin()

            # id of the row (frame) with minimum value
            curv_min_row = curv.loc[:, lower_bound:upper_bound].min(axis=1).idxmin()

        curv_max_col = curv.loc[:, int(3/4*upper_bound):int(7/4*upper_bound)].max(axis=0).idxmax()

        self.biomarkers['min'] = curv.loc[:, lower_bound:upper_bound].min().min()
        self.biomarkers['max'] = curv.loc[:, int(3/4*upper_bound):int(7/4*upper_bound)].max().max()
        self.biomarkers['min_delta'] = np.abs(curv[curv_min_col].max() - self.biomarkers['min'])
        self.biomarkers['max_delta'] = np.abs(curv[curv_max_col].min() - self.biomarkers['max'])
        self.biomarkers['amplitude_at_t'] = np.abs(curv.loc[curv_min_row].max() - self.biomarkers['min'])

        return self.biomarkers


class Cohort:

    def __init__(self, source_path='data', view='4C', output_path='data', indices_file='indices_all_cases.csv'):

        self.view = view
        self.source_path = source_path
        self.output_path = self._check_directory(output_path)
        self.indices_file = indices_file
        self.files = glob.glob(os.path.join(self.source_path, self.view, '*.CSV'))
        self.files.sort()

        self.df_all_cases = None
        self.df_master = None
        self.curv = None
        self.biomarkers = None
        self.table_name = None

    def _set_paths_and_files(self, view='4C', output_path=''):
        self.view = view
        self.files = glob.glob(os.path.join(self.source_path, self.view, '*.CSV'))
        self.files.sort()
        if not output_path=='':
            self.output_path = self._check_directory(output_path)

    @staticmethod
    def _check_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        return directory

    def _try_get_data(self, data=False, master_table=False):

        if data:
            _data_file = os.path.join(self.output_path, self.view, 'output_EDA', self.table_name)

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

            self.biomarkers = list(set([col[3:] for col in self.df_master.columns]))

        if 'label' in self.biomarkers:
            self.biomarkers.remove('label')

        if not (data or master_table):
            exit('No data has been created, set the data or master_table parameter to True')

    def _build_data_set(self, to_file=False):

        list_of_dfs = []
        for curv_file in self.files:
            print('case: {}'.format(curv_file))
            ven = Trace(curv_file, view=self.view)
            list_of_dfs.append(ven.get_biomarkers())

        self.df_all_cases = pd.concat(list_of_dfs)
        self.df_all_cases['min_index2'] = np.abs(self.df_all_cases['min_delta'] / self.df_all_cases['min'])
        self.df_all_cases['min_index'] = np.abs(self.df_all_cases['min_delta'] * self.df_all_cases['min']) * 1000
        self.df_all_cases['log_min_index'] = np.log(self.df_all_cases['min_index'])
        self.df_all_cases['min_v_amp_index'] = self.df_all_cases['min_delta'] * self.df_all_cases['amplitude_at_t']*1000
        self.df_all_cases['log_min_v_amp_index'] = np.log(self.df_all_cases['min_v_amp_index'])
        self.df_all_cases['delta_ratio'] = self.df_all_cases['min_delta'] / self.df_all_cases['max_delta']

        if to_file:
            self.df_all_cases.to_csv(os.path.join(self.output_path, self.view, 'output_EDA', self.indices_file))

    def _build_master_table(self, to_file=False, views=('4C', '3C', '2C')):

        list_of_dfs = []

        for view in views:
            self._set_paths_and_files(view=view, output_path=self.output_path)
            self._try_get_data(data=True)
            self.df_all_cases.columns = [self.view + '_' + col for col in self.df_all_cases.columns]
            list_of_dfs.append(self.df_all_cases)

        self.df_master = list_of_dfs[0]
        for df_i in range(1, len(list_of_dfs)):
            self.df_master = self.df_master.merge(list_of_dfs[df_i], right_index=True, left_index=True, sort=True)
        if to_file:
            self.df_master.to_csv(os.path.join(self.output_path, 'master_table.csv'))

    def _plot_master(self):

        if self.df_master is None:
            self._try_get_data(master_table=True)

        _master_output_path = self._check_directory(os.path.join(self.output_path, 'output_master'))
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

        _view_output_path = self._check_directory(os.path.join(self.output_path, self.view, 'output_EDA'))

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
                ven = Trace(curv_file, view=self.view)
                case_name = curv_file.split('/')[-1].split('.')[0]  # get case name without path and extension
                names[case_name] = ven.id
            if to_file:
                with open(os.path.join(self.output_path, 'Names_IDs_' + self.view + '.csv'), 'w') as f:
                    w = csv.DictWriter(f, names.keys())
                    w.writeheader()
                    w.writerow(names)

    def save_curvatures(self):

        _output_path = self._check_directory(os.path.join(self.output_path, self.view, 'output_curvature', 'curvatures'))

        for case in self.files:
            ven = Trace(case_name=case, view=self.view)
            print(ven.case_filename)
            pd.DataFrame(ven.ventricle_curvature).to_csv(os.path.join(_output_path, ven.id+'.csv'))

    def save_indices(self):

        self._build_data_set(to_file=True)

    def save_statistics(self):

        if self.df_master is None:
            self.table_name = 'master_table_with_labels.csv'
            self._try_get_data(master_table=True)

        df_stats = pd.DataFrame()
        for lab in range(3):
            df_stats['mean_'+str(lab)] = self.df_master[self.df_master['label'] == lab].mean()
        for lab in range(3):
            df_stats['std_' + str(lab)] = self.df_master[self.df_master['label'] == lab].std()
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
        _output_path = self._check_directory(os.path.join(self.output_path, self.view, 'output_EDA'))
        df_extremes.to_csv(os.path.join(_output_path, 'extremes.csv'))

    def plot_curvatures(self, coloring_scheme='curvature', plot_mean=False):

        _source_path = os.path.join(self.source_path, self.view)
        if plot_mean:
            _output_path = self._check_directory(os.path.join(self.output_path, self.view, 'output_curvature', 'mean'))
        else:
            _output_path = self._check_directory(os.path.join(self.output_path, self.view, 'output_curvature'))

        for case in self.files:
            ven = Trace(case_name=case, view=self.view, interpolation_parameters=(500, None))
            print(ven.case_filename)
            print('Points: {}'.format(ven.number_of_points))
            plot_tool = PlottingCurvature(source=_source_path,
                                          output_path=_output_path,
                                          ventricle=ven)
            if plot_mean:
                plot_tool.plot_mean_curvature()
            else:
                plot_tool.plot_all_frames(coloring_scheme=coloring_scheme)
                # ven = Trace(case_name=case, view=self.view, interpolation_parameters=(500, 500))
                # plot_tool = PlottingCurvature(source=_source_path,
                #                               output_path=_output_path,
                #                               ventricle=ven)
                plot_tool.plot_heatmap()

    def plot_distributions(self, plot_data=False, plot_master=False, table_name=None):

        self.table_name = table_name

        if plot_data:
            self._plot_data()

        if plot_master:
            self._plot_master()


class PickleReader:

    def __init__(self, source_path, output_path, model_path):
        self.source_path = source_path
        self.output_path = output_path
        self.model_path = model_path
        self.s_sopid = None
        # self.lookup_table = self._get_lookup_table()

    def _get_lookup_table(self):
        return pd.read_csv(os.path.join(self.source_path, 'image_seriessopid_lookup.csv'))

    def _get_exec_net(self):
        model_xml = os.path.join(self.model_path, 'model.xml')
        model_bin = os.path.join(self.model_path, 'model.bin')
        net = IENetwork(model=model_xml, weights=model_bin)
        plugin = IEPlugin(device="CPU")
        exec_net = plugin.load(network=net)
        del net

        return exec_net, plugin

    @staticmethod
    def _find_extreme_coordinates(mask, segment_value):
        positions = np.where(mask == segment_value)
        min_y, min_x = [np.min(p) for p in positions]
        max_y, max_x = [np.max(p) for p in positions]

        return min_x, min_y, max_x, max_y

    def _save_failed_qc_image(self, plot_title, mask):
        plt.imshow(mask)
        plt.title(plot_title)
        case_dir = check_directory(os.path.join(self.output_path, 'failed_qc',
                                                self.s_sopid.replace('.', '_')))
        shutil.rmtree('{}/*'.format(case_dir), ignore_errors=True)
        failed_dir = check_directory(os.path.join(self.output_path, 'failed_qc',
                                                  self.s_sopid.replace('.', '_'), str(self.cycle_index)))
        plt.savefig(os.path.join(failed_dir, '{}_{}_{}.png'.format(plot_title, self.cycle_index, self.img_index)))
        plt.close()

    def _check_mask_quality(self, mask):

        values, counts = np.unique(mask, return_counts=True)  # returned array is sorted

        atrium_bp_ratio = counts[3]/counts[1]
        if atrium_bp_ratio > 1.5:
            self._save_failed_qc_image('atrium_bp_raitio_ {}'.format(atrium_bp_ratio), mask)
            return False

        atrium_lv_ratio = counts[3]/(counts[1]+counts[2])
        if atrium_lv_ratio > 0.7:
            self._save_failed_qc_image('atrium_lv_ratio_ {}'.format(atrium_lv_ratio), mask)
            return False

        min_bpx, min_bpy, max_bpx, max_bpy = self._find_extreme_coordinates(mask, values[1])
        min_myox, min_myoy, max_myox, max_myoy = self._find_extreme_coordinates(mask, values[2])
        min_atx, min_aty, max_atx, max_aty = self._find_extreme_coordinates(mask, values[3])

        distances = (np.abs(min_bpx - min_myox),  # distances[0]
                     np.abs(min_bpy - min_myoy),  # distances[1]
                     np.abs(max_bpx - max_myox),  # distances[2]
                     np.abs(max_bpy - max_myoy))  # distances[3]

        if distances[0] > 35:
            self._save_failed_qc_image('delta_left_x_ {}'.format(distances[0]), mask)
            return False
        if distances[1] > 45:
            self._save_failed_qc_image('delta_lower_y_ {}'.format(distances[1]), mask)
            return False
        if distances[2] > 35:
            self._save_failed_qc_image('delta_right_x_ {}'.format(distances[2]), mask)
            return False
        if distances[3] > 15:
            self._save_failed_qc_image('delta_higher_y_ {}'.format(distances[3]), mask)
            return False

        bp_mask = np.where(mask == values[1])

        bp_above_atrium = np.sum(bp_mask[0] > min_aty)/len(bp_mask[0])  # Blood pool above the atrium mask
        if bp_above_atrium > 0.05:
            self._save_failed_qc_image('lowest_atrium_pixel_and_ratio {}_{}'.format(
                min_aty, bp_above_atrium), mask)
            return False

        return True

    def _segmentation_with_model(self, cycle_images):

        exec_net, plugin = self._get_exec_net()

        cycle_segmentations = []

        failed = 0
        for i in range(cycle_images.shape[2]):
            self.img_index = i
            img = cycle_images[:, :, i]
            img_from_array = Image.fromarray(img.astype('uint8'), 'L').transpose(Image.FLIP_LEFT_RIGHT)
            img = img_from_array.resize((256, 256))
            img_array = np.asarray(img) / 255  # Is it necessary? Scalling can be done in the line above
            img_array = cv2.pow(img_array, 0.8)
            # Plotting  # plt.imshow(img, cmap='gray')
            # plt.show()
            exec_net.start_async(request_id=0, inputs={'input_image': img_array})

            if exec_net.requests[0].wait(-1) == 0:
                net_output = exec_net.requests[0].outputs['lambda_1/Reshape_1']
                net_output = np.squeeze(net_output.transpose((2, 3, 1, 0)))
                mask = np.argmax(net_output, axis=2)
                scaling_factor = int(255 / np.max(mask))
                image_mask = Image.fromarray(scaling_factor * np.uint8(mask), mode=img.mode)
                image_mask = image_mask.resize((256, 256))
                if self._check_mask_quality(np.array(image_mask)):
                    cycle_segmentations.append(image_mask)
                else:
                    failed += 1

        del exec_net
        del plugin

        if not failed / cycle_images.shape[2] > 0.35:
            return cycle_segmentations
        else:
            return None

    @staticmethod
    def _apply_kalman(curvature):
        tran = np.eye(2 * curvature.shape[1], 2 * curvature.shape[1])
        obs = np.zeros((int(tran.shape[0] / 2), tran.shape[0]))
        init_mean = np.zeros(2 * curvature.shape[1])
        for i in range(tran.shape[1] - 1):
            print(i)
            tran[i, i + 1] = 1
            if not i % 2:
                obs[int(i / 2), i] = 1
                init_mean[i] = curvature[0][int(i / 2)]
        print(tran[:10, :10])
        print(obs[:10, :10])
        print(init_mean[:10])
        kf = KalmanFilter(transition_matrices=tran,
                          observation_matrices=obs,
                          initial_state_mean=init_mean)

        curvature, _ = kf.em(curvature).smooth(curvature)

    def _plot_relevant_cycle(self, trace):
        traces_folder = check_directory(os.path.join(self.output_path, 'traces'))
        plot_tool = PlottingCurvature(None, traces_folder, ventricle=trace)
        plot_tool.plot_all_frames(coloring_scheme='curvature')
        plot_tool.plot_heatmap()

    def _from_images_to_indices(self, cycles_list, series_uid, plot_all=False):
        """
        The main function to control the flow and call:
        1. Segmentation, which returns a list of lists (one for each cycle) of masks (one for each image).
        2. Contouring, which returns a list of lists (one for each cycle) of tuples (with ordered (x,y)
        positions of the trace.
        3. Trace, which returns the data frame with all recorded biomarkers from the provided cycles.
        :param cycles_list:
        :param series_uid:
        :return:
        """

        df_biomarkers = pd.DataFrame(columns=['min', 'max', 'min_delta', 'max_delta', 'amplitude_at_t'])

        # ----- SEGMENTATION ---------------------------------------------------------------------------------
        segmentation_list = []
        for cycle_i, cycle in enumerate(cycles_list):
            print('cycle_length: {}'.format(cycle.shape[2]))
            self.cycle_index = cycle_i
            segmentations = self._segmentation_with_model(cycle)
            if segmentations is not None:
                segmentation_list.append(segmentations)  # list of segemntations of single cycles
            else:
                print('Cycle failed on segmentation quality check')
                segmentation_list.append(None)
                seg_id = '{}_{}'.format(series_uid, cycle_i)
                df_biomarkers.loc[seg_id] = [0.0]*5

        # ----------------------------------------------------------------------------------------------------

        # ----- CONTOURING -----------------------------------------------------------------------------------
        contours_list = []
        for segment_i, segment in enumerate(segmentation_list):
            self.cycle_index = segment_i
            if segment is not None:
                contours = Contour(segmentations_path=None, output_path=self.output_path,
                                   segmentation_cycle=segment, s_sopid=self.s_sopid, cycle_index=self.cycle_index)
                contours.lv_endo_edges()
                if contours.all_cycle is not None and \
                        len(contours.all_cycle)/cycles_list[self.cycle_index].shape[2] > 0.7:
                    contours_list.append(contours.all_cycle)  # list of contours of single cycles
                else:
                    print('Cycle failed on contouring quality check')
                    contours_list.append(None)
                    seg_id = '{}_{}'.format(series_uid, segment_i)
                    df_biomarkers.loc[seg_id] = [0.0] * 5
            else:
                print('Cycle excluded due to previous segmentation failure')
                contours_list.append(None)

        # ----------------------------------------------------------------------------------------------------

        # ----- CURVATURE ------------------------------------------------------------------------------------
        traces_dict = {}
        for contour_i, contour in enumerate(contours_list):
            if contour is not None:
                trace_id = series_uid+'_'+str(contour_i)
                trace = Trace(case_name=trace_id, contours=contour,
                              interpolation_parameters=(500, None))
                traces_dict[trace_id] = trace
                df_biomarkers = df_biomarkers.append(trace.biomarkers)
        # ----------------------------------------------------------------------------------------------------

        # ----- PLOTTING -------------------------------------------------------------------------------------
        biomarkers_dir = check_directory(os.path.join(self.output_path, 'biomarkers'))
        df_biomarkers.to_csv(os.path.join(biomarkers_dir, '{}.csv'.format(series_uid)))

        print(df_biomarkers)
        min_curvature_index = self._find_trace_with_minimum_curvature(df_biomarkers[df_biomarkers])
        self._plot_relevant_cycle(traces_dict[min_curvature_index[0]])

        # TODO: change indexes and lengths (new ones are no longer valid!)
        if plot_all:
            self._plot_all(segmentation_list, cycles_list, contours_list)
        else:
            # Plotting: contour of LV on the image and mask
            min_curv_cycle = int(min_curvature_index[0].split('_')[-1])
            img_dir = check_directory(os.path.join(self.output_path, 'Seg_cont', min_curvature_index[0]))
            print('min_curv_cycle: {}'.format(min_curv_cycle))
            for i in range(len(contours_list[min_curv_cycle])):
                plt.subplot(121)
                plt.imshow(np.flip(np.array(cycles_list[min_curv_cycle][:, :, i]), axis=1), cmap='gray')
                plt.imshow(np.array(segmentation_list[min_curv_cycle][i]), cmap='YlOrBr', alpha=0.2)
                # plt.show()
                plt.subplot(122)
                plt.imshow(np.flip(np.array(cycles_list[min_curv_cycle][:, :, i]), axis=1), cmap='gray')
                plt.plot([x[0] for x in contours_list[min_curv_cycle][i]],
                         [-y[1] for y in contours_list[min_curv_cycle][i]], 'r')
                plt.savefig(os.path.join(img_dir, 'Seg_cont_{}'.format(i)))
                plt.clf()
        # ----------------------------------------------------------------------------------------------------

        print(df_biomarkers)
        return df_biomarkers

    @staticmethod
    def _find_trace_with_minimum_curvature(df):
        return df[['min']].idxmin(axis=0)

    def _plot_all(self, seg_list, cyc_list, cont_list):
        for j in range(len(seg_list)):
            for i in range(len(seg_list[j])):
                plt.subplot(121)
                plt.imshow(np.flip(np.array(cyc_list[j][:, :, i]), axis=1), cmap='gray')
                plt.imshow(np.array(seg_list[j][i]), cmap='YlOrBr', alpha=0.4)
                # plt.show()
                plt.subplot(122)
                plt.imshow(np.flip(np.array(cyc_list[j][:, :, i]), axis=1), cmap='gray')
                plt.plot([x[0] for x in cont_list[j][i]],
                         [-y[1] for y in cont_list[j][i]], 'r')
                plt.savefig(os.path.join(self.output_path, 'Seg_cont', 'Seg_cont_{}_{}'.format(j, i)))
                plt.clf()
        plt.close()

    def _print_error_file(self, cause):
        error_file_dir = check_directory(os.path.join(self.output_path, self.s_sopid))
        txt_file = open(os.path.join(error_file_dir, cause[0]), 'w')
        txt_file.write('At least one of the relevant fields in the pickle {} is missing: {}'.format(cause[0], cause[1]))
        txt_file.close()

    def _check_pickle_integrity(self, item, filename):
        item_fields = ['RDCM_viewlabel', 'time_vector', 'scanconv_movie', 'ecg_trigs']
        for field in item_fields:
            if item[field] is None:
                self._print_error_file((filename, field))
                return False
        return True

    def read_images_and_get_indices(self):
        pickles = glob.glob(os.path.join(self.source_path, '*.pck'))
        pickles = [os.path.join(self.source_path, 'CurveData_DATA_CM193022.pck')]

        list_all_biomarkers = pd.DataFrame(columns=['min', 'max', 'min_delta', 'max_delta', 'amplitude_at_t'])
        for f, filename in enumerate(pickles):  # list of the pickle files in the folder

            data = pickle.load(open(filename, 'rb'))
            print(filename)
            for s_sopid in data.keys():  # list of Series SOP instance UIDs in a pickle file
                print(s_sopid)
                self.s_sopid = s_sopid
                cycle_movies = []
                for i, item in enumerate(data[s_sopid]):  # items of Series SOP instance UID entry
                    if self._check_pickle_integrity(item, filename):
                        if item['RDCM_viewlabel'] == '4CH' and \
                                len(item['time_vector']) > 1 and \
                                item['scanconv_movie'].shape[2] > 1:
                                # and i < 6:

                            scanconv_movie = item['scanconv_movie']
                            last_cycle_triggs = item['ecg_trigs'][-2:]
                            last_cycle_frames = [np.argmin(np.abs(trig_time - item['time_vector']))
                                                 for trig_time in last_cycle_triggs]
                            cycle_movies.append(scanconv_movie[:, :,
                                                last_cycle_frames[0]:last_cycle_frames[1]+1])
                            print(i)
                            print('len time vector: {}'.format(len(item['time_vector'])))
                            print('number of frames in a movie: {}'.format(scanconv_movie.shape[2]))
                            print('ECG_TRIGS: {}'.format(item['ecg_trigs']))
                            print('last cycle trigs: {}'.format(last_cycle_triggs))
                            # Plotting              # plt.imshow(scanconv_movie[:, :, 0], cmap='gray')
                            # plt.show()

                print('cycle_movies: {}'.format(len(cycle_movies)))
                list_all_biomarkers = list_all_biomarkers.append(
                    self._from_images_to_indices(cycle_movies, s_sopid, plot_all=False))

                # TODO: Figure out what is actually being returned/saved
            # if f == 2:
        return list_all_biomarkers

    def extract_curvature_indices(self):
        list_of_biomarkers = self.read_images_and_get_indices()
        list_of_biomarkers.to_csv(os.path.join(self.output_path, 'all_biomarkers.csv'))
        # TODO: produce joint data frame and save to .csv file
        print(list_of_biomarkers)
        print(len(list_of_biomarkers))


if __name__ == '__main__':

    # Pickles
    source = os.path.join('C:/', 'Data', 'Pickles')
    output = os.path.join('c:/', 'Data', 'Pickles', '_Output')
    model = os.path.join('C:/', 'Code', 'curvature', 'model')
    pick = PickleReader(source, output, model)
    pick.extract_curvature_indices()

    # ------------------------------------------------------------------------------------------------------------------
    # Cohort
    # source = os.path.join('home','mat','Python','data','curvature')
    # target_path = os.path.join('home','mat','Python','data','curvature')
    #
    # representatives = ('RV_4C.CSV', 'CAS0214_4C.CSV', 'DPJMA0472.CSV')
    #
    # for _view in ['4C']:
    #
    #     cohort = Cohort(source_path=source, view=_view, output_path=target_path)

    # cohort.get_extemes(32)
    # cohort.plot_curvatures('asf')
    # cohort.save_curvatures()
    # cohort.plot_curvatures(coloring_scheme='curvature', plot_mean=False)
    # cohort.plot_distributions(plot_data=True, table_name='_all_cases_with_labels.csv')
    # cohort.print_names_and_ids(to_file=True)
    # cohort.get_statistics()

    # _view = '4C'
    # case_name = os.path.join(source, _view, 'AFI0442_4C.CSV')
    # ven = Ventricle(case_name, view=_view)
    # plot_tool = PlottingCurvature(source=source, output_path=target_path, ventricle=ven)
    # plot_tool.plot_representatives(representatives)
    # plot_tool.plot_mean_curvature()
    #
    # print(ven.ventricle_curvature.shape)
    # print(ven.number_of_points)
    # print(ven.number_of_frames)
    # ------------------------------------------------------------------------------------------------------------------
