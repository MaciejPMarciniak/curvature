import os
import pandas as pd
from openvino.inference_engine import IENetwork, IEPlugin
import matplotlib.pyplot as plt
import numpy as np
import shutil
from PIL import Image
import cv2
import glob
import pickle

from bsh import check_directory
from plotting import PlottingCurvature
from LV_edgedetection import Contour
from bsh import Trace


class PickleReader:

    def __init__(self, source_path, output_path, model_path):
        self.source_path = source_path
        self.output_path = output_path
        self.model_path = model_path
        self.filename = None
        self.s_sopid = None
        self.cycle_index = 0
        self.case_id = ''

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
        plt.savefig(os.path.join(failed_dir, '{}_{}_{}.png'.format(self.img_index, plot_title, self.cycle_index)))
        plt.close()

    def _check_mask_quality(self, mask):

        values, counts = np.unique(mask, return_counts=True)  # returned array is sorted

        if len(counts) != 4:
            # self._save_failed_qc_image('Chamber mask missing', mask)
            return False

        atrium_bp_ratio = counts[3]/counts[1]
        if atrium_bp_ratio > 1.5:
            # self._save_failed_qc_image('atrium_bp_raitio_ {}'.format(atrium_bp_ratio), mask)
            return False

        atrium_lv_ratio = counts[3]/(counts[1]+counts[2])
        if atrium_lv_ratio > 0.7:
            # self._save_failed_qc_image('atrium_lv_ratio_ {}'.format(atrium_lv_ratio), mask)
            return False

        min_bpx, min_bpy, max_bpx, max_bpy = self._find_extreme_coordinates(mask, values[1])
        min_myox, min_myoy, max_myox, max_myoy = self._find_extreme_coordinates(mask, values[2])
        min_atx, min_aty, max_atx, max_aty = self._find_extreme_coordinates(mask, values[3])

        distances = (np.abs(min_bpx - min_myox),  # distances[0]
                     np.abs(min_bpy - min_myoy),  # distances[1]
                     np.abs(max_bpx - max_myox),  # distances[2]
                     np.abs(max_bpy - max_myoy),  # distances[3]
                     np.abs(max_myoy - min_aty))  # distances[4]

        if distances[0] > 35:
            # self._save_failed_qc_image('delta_left_x_ {}'.format(distances[0]), mask)
            return False
        if distances[1] > 35:
            # self._save_failed_qc_image('delta_lower_y_ {}'.format(distances[1]), mask)
            return False
        if distances[2] > 35:
            # self._save_failed_qc_image('delta_right_x_ {}'.format(distances[2]), mask)
            return False
        if distances[3] > 15:
            # self._save_failed_qc_image('delta_higher_y_ {}'.format(distances[3]), mask)
            return False
        if distances[4] > 20:
            # self._save_failed_qc_image('delta_higher_y_ {}'.format(distances[3]), mask)
            return False

        bp_mask = np.where(mask == values[1])

        bp_above_atrium = np.sum(bp_mask[0] > min_aty)/len(bp_mask[0])  # Blood pool above the atrium mask
        if bp_above_atrium > 0.07:
            # self._save_failed_qc_image('lowest_atrium_pixel_and_ratio {}'.format(bp_above_atrium), mask)
            return False

        if min_bpy < 5:
            # self._save_failed_qc_image('blood pool at the edge of image', mask)
            return False

        return True

    def _segmentation_with_model(self, cycle_images, exec_net, plugin):

        cycle_segmentations = []

        failed = 0
        failed_ids = []
        for i in range(cycle_images.shape[2]):
            self.img_index = i
            img = cycle_images[:, :, i]
            img_from_array = Image.fromarray(img.astype('uint8'), 'L').transpose(Image.FLIP_LEFT_RIGHT)
            img_from_array = img_from_array.resize((256, 256))
            img_array = np.asarray(img_from_array) / 255  # Is it necessary?
            img_array = cv2.pow(img_array, 0.8)
            # Plotting  # plt.imshow(img, cmap='gray')
            # plt.show()
            exec_net.start_async(request_id=0, inputs={'input_image': img_array})

            if exec_net.requests[0].wait(-1) == 0:
                net_output = exec_net.requests[0].outputs['lambda_1/Reshape_1']
                net_output = np.squeeze(net_output.transpose((2, 3, 1, 0)))
                mask = np.argmax(net_output, axis=2)
                scaling_factor = int(255 / np.max(mask))
                image_mask = Image.fromarray(scaling_factor * np.uint8(mask), mode=img_from_array.mode)
                image_mask = image_mask.resize((256, 256))
                if self._check_mask_quality(np.array(image_mask)):
                    cycle_segmentations.append(image_mask)
                else:
                    # failed_dir = check_directory(os.path.join(self.output_path, 'failed_qc',
                    #                                           self.s_sopid.replace('.', '_'), str(self.cycle_index)))
                    # img_from_array.save(os.path.join(failed_dir, 'failed_{}.png'.format(self.img_index)))
                    failed += 1
                    failed_ids.append(i)

        if not failed / cycle_images.shape[2] > 0.35:
            return cycle_segmentations, failed_ids
        else:
            return None, None

    def _plot_relevant_cycle(self, trace):
        traces_folder = check_directory(os.path.join(self.output_path, 'traces'))
        plot_tool = PlottingCurvature(None, traces_folder, ventricle=trace)
        plot_tool.plot_heatmap()
        plot_tool.plot_all_frames(coloring_scheme='curvature')

    def _find_and_save_ed(self, segmentations, cycle_id):

        bp_counts = []
        for seg_i, seg in enumerate(segmentations):
            values, counts = np.unique(seg, return_counts=True)  # returned array is sorted
            bp_counts.append(counts[1])
        ed_id = np.argmax(bp_counts)
        plt.imshow(segmentations[ed_id], cmap='gray')
        plt.savefig(os.path.join(self.output_path, 'EDs', '{}_{}.png'.format(self.case_id, cycle_id)))
        plt.close()

    def _from_images_to_indices(self, cycles_list, dimensions_list, plot_all=False):
        """
        The main function to control the flow and call:
        1. Segmentation, which returns a list of lists (one for each cycle) of masks (one for each image).
        2. Contouring, which returns a list of lists (one for each cycle) of tuples (with ordered (x,y)
        positions of the trace.
        3. Trace, which returns the data frame with all recorded biomarkers from the provided cycles.
        :param cycles_list:
        :return:
        """

        df_biomarkers = pd.DataFrame(columns=['min', 'max', 'avg_min_basal_curv', 'avg_avg_basal_curv',
                                              'min_delta', 'max_delta', 'amplitude_at_t'])

        # ----- SEGMENTATION ---------------------------------------------------------------------------------
        segmentation_list = []
        df = pd.DataFrame(index=['Patient_ID'], columns=['x', 'y'])
        exec_net, plugin = self._get_exec_net()
        for cycle_i, cycle in enumerate(cycles_list):
            print('cycle_length: {}'.format(cycle.shape[2]))
            self.cycle_index = cycle_i
            segmentations, _ = self._segmentation_with_model(cycle, exec_net, plugin)
            if segmentations is not None:
                self._find_and_save_ed(segmentations, cycle_i)
                df.loc['{}_{}'.format(self.case_id, cycle_i)] = dimensions_list[cycle_i]
                segmentation_list.append(segmentations)  # list of segemntations of single cycles
            else:
                print('Cycle failed on segmentation quality check')
                segmentation_list.append(None)
                seg_id = '{}_{}'.format(self.s_sopid, cycle_i)
                df_biomarkers.loc[seg_id] = [0.0] * 7
        del exec_net
        del plugin
        df.to_csv(os.path.join(self.output_path, 'EDs', '{}.csv'.format(self.case_id)))
        # ----------------------------------------------------------------------------------------------------

        # ----- CONTOURING -----------------------------------------------------------------------------------
        contours_list = []
        for segment_i, segment in enumerate(segmentation_list):
            self.cycle_index = segment_i
            if segment is not None:
                contours = Contour(segmentations_path=None, output_path=self.output_path,
                                   segmentation_cycle=segment, s_sopid=self.s_sopid,
                                   cycle_index=self.cycle_index, dimensions=dimensions_list[segment_i])
                contours.lv_endo_edges()

                if contours.all_cycle is not None and \
                        len(contours.all_cycle)/cycles_list[self.cycle_index].shape[2] > 0.7:
                    contours_list.append(contours.all_cycle)  # list of contours of single cycles
                else:
                    print('Cycle failed on contouring quality check')
                    contours_list.append(None)
                    seg_id = '{}_{}'.format(self.s_sopid, segment_i)
                    df_biomarkers.loc[seg_id] = [1.0] * 7
            else:
                print('Cycle excluded due to previous segmentation failure')
                contours_list.append(None)

        # ----------------------------------------------------------------------------------------------------

        # ----- CURVATURE ------------------------------------------------------------------------------------
        traces_dict = {}
        for contour_i, contour in enumerate(contours_list):
            if contour is not None:
                trace_id = self.s_sopid + '_'+str(contour_i)
                trace = Trace(case_name=trace_id, contours=contour,
                              interpolation_parameters=(None, False))
                traces_dict[trace_id] = trace
                np.savetxt(os.path.join(self.output_path,
                                        'Curvatures', '{}_{}.csv'.format(self.case_id, str(contour_i))),
                           np.array(trace.ventricle_curvature), delimiter=',', fmt='%.7f')
                np.savetxt(os.path.join(self.output_path,
                                        'Contours', '{}_{}.csv'.format(self.case_id, str(contour_i))),
                           trace.data, delimiter=',', fmt='%.7f')
                df_biomarkers = df_biomarkers.append(trace.biomarkers)
        # ----------------------------------------------------------------------------------------------------

        # ----- PLOTTING -------------------------------------------------------------------------------------
        # biomarkers_dir = check_directory(os.path.join(self.output_path, 'biomarkers'))
        # df_biomarkers.to_csv(os.path.join(biomarkers_dir, '{}.csv'.format(series_uid)))
        df_tmp = df_biomarkers[df_biomarkers['min'] != 0]
        df_tmp = df_tmp[df_tmp['min'] != 1]
        if df_tmp.empty:
            self._print_error_file_cycles()
            return df_biomarkers
        # Plotting: the trace with maximum curvature of the case
        min_curvature_index = self._find_trace_with_minimum_curvature(df_biomarkers)

        self._plot_relevant_cycle(traces_dict[min_curvature_index[0]])

        if plot_all:
            self._plot_all(segmentation_list, cycles_list, contours_list)
        else:
            # Plotting: contour of LV on the image and mask
            min_curv_cycle = int(min_curvature_index[0].split('_')[-1])
            img_dir = check_directory(os.path.join(self.output_path,
                                                   'Seg_cont',
                                                   '{}_{}'.format(self.filename, self.case_id)))
            print('min_curv_cycle: {}'.format(min_curv_cycle))
            for i in range(len(contours_list[min_curv_cycle])):
                plt.subplot(121)
                plt.imshow(np.flip(np.array(cycles_list[min_curv_cycle][:, :, i]), axis=1), cmap='gray')
                plt.imshow(np.array(segmentation_list[min_curv_cycle][i]), cmap='jet', alpha=0.2)
                plt.subplot(122)
                plt.imshow(np.flip(np.array(cycles_list[min_curv_cycle][:, :, i]), axis=1), cmap='gray')
                plt.plot([x[0]/dimensions_list[min_curv_cycle][0] for x in contours_list[min_curv_cycle][i]],
                         [-y[1]/dimensions_list[min_curv_cycle][1] for y in contours_list[min_curv_cycle][i]],
                         'y--')
                plt.savefig(os.path.join(img_dir, 'Seg_cont_{}'.format(i)))
                plt.clf()
        # ----------------------------------------------------------------------------------------------------

        print(df_biomarkers)
        return df_biomarkers

    @staticmethod
    def _find_trace_with_minimum_curvature(df):
        return df[['avg_min_basal_curv']].idxmin(axis=0)

    @staticmethod
    def _get_width_and_height_scales(start_angle, stop_angle, start_depth, stop_depth, resolution=256):
        """
        Python implementation of the matlab code in idunn/matlab/UQTools/common.
        expects image as a numpy array.
        """
        angle_increment = (stop_angle - start_angle) / (resolution - 1)

        # rotate 90 degrees clockwise to adapt with old EchoPAC angle defs
        start_angle = start_angle + np.pi / 2
        stop_angle = stop_angle + np.pi / 2

        # set size of image if not defined
        angle_range = np.arange(start_angle, stop_angle, angle_increment)

        xmin = -1 * np.max(np.cos(start_angle % np.pi) * np.array([start_depth, stop_depth]))
        xmax = -1 * np.min(np.cos(stop_angle % np.pi) * np.array([start_depth, stop_depth]))
        ymin = np.min(np.sin(angle_range % np.pi) * start_depth)
        ymax = np.max(np.sin(angle_range % np.pi) * stop_depth)

        height_scaler = (ymax - ymin) * 1000 / resolution
        width_scaler = (xmax - xmin) * 1000 / resolution

        return height_scaler, width_scaler

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

    def _print_error_file_pickle(self, cause):
        print('Pickle fields missing')
        error_filename = cause[0].split('.')[0]
        error_file_dir = check_directory(os.path.join(self.output_path, 'failed_qc', error_filename))
        error_file = os.path.join(error_file_dir, '{}.txt'.format(error_filename))
        txt_file = open(error_file, 'w+')
        txt_file.write('At least one of the relevant fields in the pickle {} is missing: {}'.format(cause[0],
                                                                                                    cause[1]))
        txt_file.close()

    def _print_error_file_cycles(self):
        print('No usable cycles found')
        error_filename = self.filename.split('.')[0]
        error_file_dir = check_directory(os.path.join(self.output_path, 'failed_qc', 'failed_cycles'))
        txt_file = open(os.path.join(error_file_dir, '{}.txt'.format(self.s_sopid)), 'w')
        txt_file.write('No usable cycles found in case {}'.format(self.s_sopid))
        txt_file.close()
        txt_file2 = open(os.path.join(error_file_dir, '{}.txt'.format(error_filename)), 'w')
        txt_file2.write('No usable cycles found in case {}'.format(error_filename))
        txt_file2.close()

    def _print_error_file_corrupted(self):
        print('Pickle file corrupted')
        filename = self.filename.split('.')[0]
        error_file_dir = check_directory(os.path.join(self.output_path, 'failed_qc', filename))
        error_file = os.path.join(error_file_dir, '{}.txt'.format(filename))
        txt_file = open(error_file, 'w')
        txt_file.write('File {} is corrupted'.format(filename))
        txt_file.close()

    def get_biomakers(self):
        curves = glob.glob(os.path.join(self.output_path, 'Curvatures', '*.csv'))
        curves.sort()
        for curve in curves:
            df_curve = pd.read_csv(curve)
            df_curve = df_curve.mean(axis=0)
            df_curve = df_curve.iloc[20:150]

    def _check_pickle_integrity(self, item, filename):
        item_fields = ['RDCM_viewlabel', 'time_vector', 'scanconv_movie', 'ecg_trigs']
        for field in item_fields:
            if item[field] is None:
                self._print_error_file_pickle((filename, field))
                return False

        if not len(item['time_vector']) > 1 and \
                item['scanconv_movie'].shape[2] > 1 and \
                item['params']['depth_start'] is not None and \
                item['params']['depth_end'] is not None and \
                item['params']['vector_angles'] is not None:
            return False

        return True

    def read_images_and_get_indices(self, get_ed=False):
        pickles = glob.glob(os.path.join(self.source_path, '*.pck'))
        pickles.sort()
        # df_bsh = list(pd.read_csv(os.path.join(self.source_path, 'bsh_sop.csv'), header=0))
        # print(df_bsh)
        df_all_biomarkers = pd.DataFrame(columns=['min', 'max', 'avg_min_basal_curv', 'avg_avg_basal_curv',
                                                  'min_delta', 'max_delta', 'amplitude_at_t',
                                                  'Series_SOP', 'Patient_ID'])

        for f, filename in enumerate(pickles):  # list of the pickle files in the folder
            self.filename = filename
            try:
                data = pickle.load(open(filename, 'rb'))
            except pickle.UnpicklingError:
                self._print_error_file_corrupted()
                continue
            except EOFError:
                self._print_error_file_corrupted()
                continue

            print(filename)
            self.s_sopid = list(data.keys())[0]  # list of Series SOP instance UIDs in a pickle file

            cycle_movies = []
            cycle_dimensions = []
            for i, item in enumerate(data[self.s_sopid]):  # items of Series SOP instance UID entry
                if self._check_pickle_integrity(item, filename) and \
                        item['RDCM_viewlabel'] == '4CH' and i < 4:

                    self.case_id = item['patient_id']

                    image_parameters = [item['params']['vector_angles'][0],
                                        item['params']['vector_angles'][-1],
                                        item['params']['depth_start'], item['params']['depth_end']]

                    # Dimensions of the image
                    height_mm_scale, width_mm_scale = self._get_width_and_height_scales(*image_parameters)

                    # Entire acquisition of the image
                    scanconv_movie = item['scanconv_movie']

                    # Frames close to ECG markers of the cycles
                    cycle_frames = [np.argmin(np.abs(trig_time - item['time_vector']))
                                    for trig_time in item['ecg_trigs']]

                    # Separated movies of the cycles
                    for frame in range(len((cycle_frames[:-1]))):
                        # for frame in range(len((cycle_frames[-3:-2]))):  # testing
                        cycle_movies.append(scanconv_movie[:, :,
                                            cycle_frames[frame]:cycle_frames[frame+1]])
                        cycle_dimensions.append((width_mm_scale, height_mm_scale))

                    print('len time vector: {}'.format(len(item['time_vector'])))
                    print('cycle_frames: {}'.format(cycle_frames))
                    print('Scaling factors: width - {}, height - {}'.format(width_mm_scale, height_mm_scale))

            print('cycle_movies: {}'.format(len(cycle_movies)))

            if len(cycle_movies) > 0:
                tmp_biomarkers = self._from_images_to_indices(cycle_movies, cycle_dimensions,
                                                              plot_all=False)
                tmp_biomarkers.loc[:, 'Series_SOP'] = self.s_sopid
                tmp_biomarkers.loc[:, 'Patient_ID'] = self.case_id
                df_all_biomarkers = df_all_biomarkers.append(tmp_biomarkers, sort=False)
                df_all_biomarkers.to_csv(os.path.join(self.output_path, 'biomarkers',
                                                      '{}.csv'.format(self.case_id)))
            else:
                self._print_error_file_corrupted()

        return df_all_biomarkers

    @staticmethod
    def _crop_image(image, mask, dimensions, border=None):

        if border is None:
            myo = np.where(mask == 170)
            base = np.max(myo[0]) + 10 if np.max(myo[0]) + 10 < 255 else 255
            apex = np.min(myo[0]) - 5 if np.min(myo[0]) - 5 > 0 else 0
            center = int(np.mean(np.where(mask == 85)[1]))
            crop_len = base - apex
            left, right = center - int(crop_len / 2), center + int(crop_len / 2)
            border = [apex, base, left, right]

        vertical_len = border[1] - border[0]
        horizontal_len = border[3] - border[2]
        cropped_image = image[border[0]:border[1], border[2]:border[3]]
        cropped_image = cv2.resize(np.array(cropped_image), (256, 256))
        cropped_dimensions = np.array([dimensions[0] * vertical_len, dimensions[1] * horizontal_len]) / 256

        # plt.subplot(121)
        # plt.imshow(image, cmap='gray')
        # plt.scatter(120, base, color='r')
        # plt.scatter(120, apex, color='g')
        # plt.scatter(center,120, color='b')
        # plt.subplot(122)
        # plt.imshow(cropped_image, cmap='gray')
        # plt.show()
        # plt.close()

        return cropped_image, cropped_dimensions, border

    def _crop_cycle(self, cycle_list, dimensions_list):

        cropped_cycles = []
        cropped_dimensions = []
        crp_dim = None
        exec_net, plugin = self._get_exec_net()
        for cycle_i, cycle in enumerate(cycle_list):
            self.cycle_index = cycle_i
            segmentations, _ = self._segmentation_with_model(cycle, exec_net, plugin)

            if segmentations is not None:
                crp_cycle = np.zeros((256, 256, len(segmentations)))
                border = None
                for m, mask in enumerate(segmentations):
                    mask = cv2.resize(np.array(mask), (256, 256))
                    crp_im, crp_dim, border = self._crop_image(cycle[:, :, m], np.fliplr(mask),
                                                               dimensions_list[cycle_i], border)
                    crp_cycle[:, :, m] = crp_im

                cropped_dimensions.append(crp_dim)
                cropped_cycles.append(crp_cycle)  # list of cropped images in cycles

        del exec_net
        del plugin

        return cropped_cycles, cropped_dimensions

    def _get_segmentations(self, cycle_list, dimensions_list):

        segmented_cycles = []
        segmented_dimensions = []
        failed_frames = []
        exec_net, plugin = self._get_exec_net()
        for cycle_i, cycle in enumerate(cycle_list):
            self.cycle_index = cycle_i
            segmentations, failed = self._segmentation_with_model(cycle, exec_net, plugin)

            if segmentations is not None:
                segmented_cycles.append(segmentations)
                segmented_dimensions.append(dimensions_list[cycle_i])  # list of cropped images in cycles
                failed_frames.append(failed)
            else:
                segmented_cycles.append([-1])
                segmented_dimensions.append([-1])  # list of cropped images in cycles
                failed_frames.append([-1])

        del exec_net
        del plugin

        return segmented_cycles, segmented_dimensions, failed_frames

    def _save_cycles(self, cycles, failed_frames=(None), resize=True, sequence='', subject='', kind='bsh_examples'):

        out_dir = check_directory(os.path.join(self.output_path, kind))
        out_dir = check_directory(os.path.join(out_dir, '{}'.format(subject)))

        for ci, cycle in enumerate(cycles):
            print(ci)
            if len(cycle) == 1:
                print('Single image {}'.format(self.filename))
                continue
            cycle_out_dir = check_directory(os.path.join(out_dir, 'Sequence_{} {}'.format(sequence, ci)))
            if isinstance(cycle, list):
                # for ffi in failed_frames[ci]:
                #     cycle.insert(ffi, np.zeros((256, 256)))
                for frame in range(len(cycle)):
                    # if frame in failed_frames[ci]:
                    #     continue
                    image = np.array(cycle[frame])
                    cv2.imwrite(os.path.join(cycle_out_dir, 'US_2D_{}.png'.format(frame)), image)
            else:
                for frame in range(cycle.shape[2]):
                    # if frame in failed_frames[ci]:
                    #     continue
                    image = np.fliplr(cycle[:, :, frame])
                    if resize:
                        image = cv2.resize(np.array(image), (1024, 1024), interpolation=cv2.INTER_LINEAR_EXACT)
                    cv2.imwrite(os.path.join(cycle_out_dir, 'US_2D_{}.png'.format(frame)), image)

    def save_relevant_images(self):
        pickles = glob.glob(os.path.join(self.source_path, '*.pck'))
        pickles.sort()
        # df_pics = pd.read_excel('rel_cyc.xlsx', index_col=0, header=0)
        s = 0
        for f, filename in enumerate(pickles):  # list of the pickle files in the folder
            self.filename = filename.split('\\')[-1].split('.')[0]
            print(self.filename)
            print(f)
            # if 'AduHeart' in self.filename:
            #     continue
            try:
                data = pickle.load(open(filename, 'rb'))
            except pickle.UnpicklingError:
                self._print_error_file_corrupted()
                continue
            except EOFError:
                self._print_error_file_corrupted()
                continue

            for s_sopid in data.keys():  # list of Series SOP instance UIDs in a pickle file
                print(s_sopid)
                self.s_sopid = s_sopid

                # if self.s_sopid not in df_pics['sop'].values:
                #     print('not saved: {}'.format(self.s_sopid))
                #     break

                for i, item in enumerate(data[s_sopid]):  # items of Series SOP instance UID entry
                    cycle_movies = []
                    cycle_dimensions = []
                    sequence_list = []
                    if self._check_pickle_integrity(item, filename):  # and \
                            # item['RDCM_viewlabel'] == '4CH':  # and i < 4:

                        sequence_list.append(i)
                        self.case_id = item['patient_id']
                        print('saving {}'.format(self.case_id))
                        # image_parameters = [item['params']['vector_angles'][0],
                        #                     item['params']['vector_angles'][-1],
                        #                     item['params']['depth_start'], item['params']['depth_end']]
                        s+=1
                        print(s)
                        # Dimensions of the image
                        # height_mm_scale, width_mm_scale = self._get_width_and_height_scales(*image_parameters)

                        # Entire acquisition of the image
                        scanconv_movie = item['scanconv_movie']

                        # Frames close to ECG markers of the cycles
                        # cycle_frames = [np.argmin(np.abs(trig_time - item['time_vector']))
                        #                 for trig_time in item['ecg_trigs']]
                        cycle_movies.append(scanconv_movie)
                        # Separated movies of the cycles
                        # for frame in range(len((cycle_frames[:-1]))):
                            # for frame in range(len((cycle_frames[-3:-2]))):  # testing
                            # cycle_movies.append(scanconv_movie[:, :,
                            #                     cycle_frames[frame]:cycle_frames[frame + 1]])
                            # cycle_dimensions.append((width_mm_scale, height_mm_scale))
                        # print(len(cycle_movies))
                        # segmentations, seg_cd, seg_failed = self._get_segmentations(cycle_movies,
                        #                                                             cycle_dimensions)
                        # print(len(segmentations))
                        # print(len(seg_failed))
                        self._save_cycles(cycle_movies, failed_frames=(), subject=self.filename,
                                          sequence=str(sequence_list[-1]), resize=True, kind='Images')
                        # self._save_cycles(segmentations, failed_frames=seg_failed, subject=self.filename,
                        #                   sequence=str(sequence_list[-1]), resize=False, kind='Segmentations')

    def extract_curvature_indices(self):
        list_of_biomarkers = self.read_images_and_get_indices()
        list_of_biomarkers.index.name = 'cycle_id'
        list_of_biomarkers.to_csv(os.path.join(self.output_path, 'all_biomarkers.csv'))
        print(list_of_biomarkers)
        print(len(list_of_biomarkers))


if __name__=='__main__':
    # Pickles
    source = os.path.join('C:/', 'Data', 'PaduaSB')
    output = os.path.join('F:/', 'export')
    model = os.path.join('C:/', 'Code', 'curvature', 'model')
    pick = PickleReader(source, output, model)
    pick.save_relevant_images()

    # pick.extract_curvature_indices()
    # pick.get_biomakers()