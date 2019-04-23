import os
import glob
from ntpath import basename
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import cv2
import shutil


def check_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


class Contour:

    def __init__(self, segmentations_path, output_path, segmentation_cycle, s_sopid, cycle_index):
        self.segmentations_path = segmentations_path
        self.output_path = self._check_directory(output_path)
        self.segmentation_cycle = segmentation_cycle
        self.s_sopid = s_sopid
        self.cycle_index = cycle_index
        self.img_index = 0
        if self.segmentations_path is not None:
            self.seg_files = glob.glob(os.path.join(self.segmentations_path, '*.png'))
            self.seg_files.sort()
        self.current_gray_mask = None
        self.sorted_edge = list()
        self.all_cycle = None

    @staticmethod
    def _check_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        return directory

    def _correct_indices(self):
        # To make sure that endocardium and not the cavity is captured, relevant indices are moved by 1
        row_diffs = np.diff(self.current_gray_mask, axis=1)
        row_diffs_corr = np.where(row_diffs == 85)
        row_diffs_corr = (row_diffs_corr[0], row_diffs_corr[1] + 1)  # index correction
        col_diffs = np.diff(self.current_gray_mask, axis=0)
        col_diffs_corr = np.where(col_diffs == 85)
        col_diffs_corr = (col_diffs_corr[0] + 1, col_diffs_corr[1])  # index correction

        edge = list()
        edge.append(np.where(row_diffs == 256 - 85))
        edge.append(row_diffs_corr)
        edge.append(np.where(col_diffs == 256 - 85))
        edge.append(col_diffs_corr)

        return edge

    def _extract_edge_image(self):
        _lv_edge = np.zeros(self.current_gray_mask.shape, dtype=np.uint8)
        corrected_indices = self._correct_indices()
        for edge_pixels in corrected_indices:
            _lv_edge[edge_pixels] = 255

        return _lv_edge

    @staticmethod
    def _pair_coordinates(edge):
        indices = np.where(edge == 255)
        y_pos = -indices[0]
        x_pos = indices[1]
        return np.array([(x, y) for x, y in zip(x_pos, y_pos)])

    def _look_around(self, coordinates_of_edge, cur_point, previous_point):

        coordinates_of_edge = coordinates_of_edge.tolist()
        # TODO: Use hash tables for faster checking
        touching_points = list()
        for i in [cur_point[0]-1, cur_point[0], cur_point[0]+1]:
            for j in [cur_point[1]-1, cur_point[1], cur_point[1]+1]:
                if [i, j] in coordinates_of_edge and \
                        np.all((i, j) != cur_point) and \
                        np.all((i, j) != previous_point):
                    touching_points.append((i, j))

        if len(touching_points) == 1:  # If only one touching point was found return
            return touching_points[0], False

        for point in touching_points:  # If more points were found, return diagonal one
            if point[0] != cur_point[0] and point[1] != cur_point[1] and point not in self.sorted_edge:
                return point, False

        for point in touching_points:
            # Due to segmentation faults, there are special cases - then choose point further from the previous point
            if point[0] != previous_point[0] and point[1] != previous_point[1] \
                    and point not in self.sorted_edge:
                return point, False

        return cur_point, True

    @staticmethod
    def smooth(x, window_len=15, window='hanning'):
        """smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal

        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        """

        if x.ndim != 1:
            exit("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            exit("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            exit("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    def _walk_on_edge(self, coordinates_of_edge):
        """
        Since the ventricle is usually not convex, using radial coordinates can be misleading. A simple search
        deals with the proper order of the points and ensures a single-pixel edge.
        :param coordinates_of_edge: list of (x, y) coordinates of the edge on 2D plane
        :return: sorted list of coordinates of the edge
        """
        self.sorted_edge = list()
        edge_points = list()
        cur_point = tuple(max(coordinates_of_edge, key=lambda x: x[1]))
        prev_point = cur_point
        self.sorted_edge.append(cur_point)
        while 1:
            try:
                new_point, flag = self._look_around(coordinates_of_edge, cur_point, prev_point)
            except TypeError:
                plt.scatter(self.sorted_edge, s=1)
                plt.xlim((0, 256))
                plt.ylim((-256, 0))
                plt.show()
                exit()

            if flag:
                edge_points.append(cur_point)
                if len(edge_points) == 2:
                    break
                self.sorted_edge.reverse()
                cur_point = self.sorted_edge[-1]
                prev_point = self.sorted_edge[-2]
            else:
                prev_point = cur_point
                cur_point = new_point
                self.sorted_edge.append(cur_point)

        basal_septal_edge = min(edge_points, key=lambda x: x[0])
        if basal_septal_edge != self.sorted_edge[0]:
            self.sorted_edge.reverse()
        se_x = self._expand_edge(np.array([x[0] for x in self.sorted_edge]))
        se_y = self._expand_edge(np.array([y[1] for y in self.sorted_edge]))
        interp_se_x = self.smooth(se_x)[7:-7]
        interp_se_y = self.smooth(se_y)[7:-7]

        # plt.scatter(range(len(se_x)), se_x)
        # plt.plot(interp_se_x, 'r', lw=3)

        smooth_se_x = savgol_filter(interp_se_x, 11, polyorder=3, mode='interp')[13:-13]
        smooth_se_y = savgol_filter(interp_se_y, 11, polyorder=3, mode='interp')[13:-13]

        # plt.plot(range(13, len(smooth_se_x)+13), smooth_se_x, 'g')
        # plt.show()

        self.sorted_edge = [(x, y) for x, y in zip(smooth_se_x, smooth_se_y)]

        return self.sorted_edge

    @staticmethod
    def _expand_edge(sorted_edge_axis):
        beg_ = np.ones(10) * sorted_edge_axis[0]
        end_ = np.ones(10) * sorted_edge_axis[-1]
        return np.concatenate((beg_, sorted_edge_axis, end_))

    def _save_results(self, cur_edge, basename_file):
        out_dir = self._check_directory(os.path.join(self.output_path, 'Contour_tables'))
        np.savetxt(os.path.join(out_dir, basename_file + '.csv'), self.sorted_edge)
        plt.imshow(cur_edge, cmap='gray')
        out_dir = self._check_directory(os.path.join(self.output_path, 'Edge_images'))
        plt.savefig(os.path.join(out_dir, basename_file + '.png'))
        plt.clf()
        self.sorted_edge = np.array(self.sorted_edge)
        plt.plot(self.sorted_edge[:, 0], self.sorted_edge[:, 1], 'k-')
        plt.xlim((0, 256))
        plt.ylim((-256, 0))
        plt.savefig(os.path.join(out_dir, basename_file + '_ordered.png'))
        plt.clf()

    def _lv_endo_edges(self, seg_mask):

        if np.any(seg_mask.size != (256, 256)):
            print(seg_mask.size)
            seg_mask = cv2.resize(np.array(seg_mask), (256, 256))

        if seg_mask.size == (256, 256, 3):
            seg_mask_gray = cv2.cvtColor(np.array(seg_mask), cv2.COLOR_BGR2GRAY)
        else:
            seg_mask_gray = np.array(seg_mask)
        # Plotting # plt.imshow(seg_mask)
        # plt.show()
        self.current_gray_mask = seg_mask_gray
        current_lv_edge = self._extract_edge_image()
        coord_lv = self._pair_coordinates(current_lv_edge)
        coord_lv_ordered = self._walk_on_edge(coord_lv)
        if len(coord_lv_ordered) < 100:
            plt.subplot(121)
            plt.imshow(seg_mask)
            plt.subplot(122)
            plt.imshow(current_lv_edge)
            plt.show()

        return coord_lv_ordered

    def _check_contour_quality(self, prev_cont, mask):

        percent_prev_contour_diff = np.abs(len(self.sorted_edge) - len(prev_cont)) / len(prev_cont)
        if percent_prev_contour_diff > 0.25:  # 25% of previous contour
            self._save_failed_qc_image('percent_prev_cont {}'.format(percent_prev_contour_diff), mask)
            return False

        values, counts = np.unique(mask, return_counts=True)  # returned array is sorted

        positions = np.where(mask == values[1])
        max_bp_y, max_bp_x = [np.max(p) for p in positions]
        min_bp_y, min_bp_x = [np.min(p) for p in positions]
        max_cont_y = np.max([-c[1] for c in self.sorted_edge])
        min_cont_y = np.min([-c[1] for c in self.sorted_edge])
        max_cont_x = np.max([c[0] for c in self.sorted_edge])
        min_cont_x = np.min([c[0] for c in self.sorted_edge])

        if max_bp_y < max_cont_y - 0.6:
            self._save_failed_qc_image('Contour over bp', mask)
            return False

        diff_max_y = np.abs(max_bp_y - max_cont_y)
        if diff_max_y > 10:
            self._save_failed_qc_image('Diff_max_bp_y_max_cont_y {}'.format(diff_max_y), mask)
            return False

        diff_min_y = np.abs(min_bp_y - min_cont_y)
        if diff_min_y > 15:
            self._save_failed_qc_image('Diff_min_bp_y_min_cont_y {}'.format(diff_min_y), mask)
            return False

        diff_max_x = np.abs(max_bp_x - max_cont_x)
        if diff_max_x > 15:
            self._save_failed_qc_image('Diff_max_bp_x_max_cont_x {}'.format(diff_max_x), mask)
            return False

        diff_min_x = np.abs(min_bp_x - min_cont_x)
        if diff_min_x > 15:
            self._save_failed_qc_image('Diff_min_bp_x_min_cont_x {}'.format(diff_min_x), mask)
            return False

        return True

    def _save_failed_qc_image(self, plot_title, mask):
        plt.imshow(mask)
        plt.plot([x[0] for x in self.sorted_edge], [-y[1] for y in self.sorted_edge], 'r')
        plt.title(plot_title)
        case_dir = check_directory(os.path.join(self.output_path, 'failed_qc',
                                                self.s_sopid.replace('.', '_')))
        shutil.rmtree('{}/*'.format(case_dir), ignore_errors=True)
        failed_dir = check_directory(os.path.join(self.output_path, 'failed_qc',
                                                  self.s_sopid.replace('.', '_'), str(self.cycle_index)))
        plt.savefig(os.path.join(failed_dir, '{}_{}_{}.png'.format(self.img_index, plot_title,
                                                                   self.cycle_index)))
        plt.close()

    def lv_endo_edges(self):

        if self.segmentations_path is not None:
            for seg_file in self.seg_files:
                print(seg_file)
                seg_mask = cv2.imread(seg_file)
                endo_coords = self._lv_endo_edges(seg_mask)
                self._save_results(endo_coords, basename(seg_file)[:-4])

        elif self.segmentation_cycle is not None:

            print('length of the cycle: {}'.format(len(self.segmentation_cycle)))
            cycle_coords = []
            self._lv_endo_edges(self.segmentation_cycle[0])
            prev_cont = self.sorted_edge

            failed = 0
            for seg_img_i, seg_img in enumerate(self.segmentation_cycle):
                endo_coords = self._lv_endo_edges(seg_img)
                self.img_index = seg_img_i
                # print('len of edge: {}'.format(len(self.sorted_edge)))
                if self._check_contour_quality(prev_cont, np.array(seg_img)):
                    cycle_coords.append(endo_coords[::3])
                    prev_cont = self.sorted_edge
                else:
                    failed += 1

            if not failed / len(self.segmentation_cycle) > 0.35:
                self.all_cycle = cycle_coords
            else:
                self.all_cycle = None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create contours to input into the curvature model')
    parser.add_argument('-s', '--segmentations', help='Segmentation results from NTNU model', required=True)
    parser.add_argument('-o', '--output_path', help='Directory where output will be stored', required=True)
    args = parser.parse_args()
    cont = Contour(args.segmentations, args.output_path)
    cont.lv_endo_edges()
