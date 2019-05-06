import os
import glob
from ntpath import basename
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf
from scipy.interpolate import UnivariateSpline
import cv2
import shutil


def check_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


class Contour:

    def __init__(self, segmentations_path, output_path, segmentation_cycle=None, s_sopid=None, cycle_index=None):
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
        self.endo_sorted_edge = list()
        self.epi_sorted_edge = list()
        self.all_cycle = None
        self.endo = True


    @staticmethod
    def _check_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        return directory

    @staticmethod
    def _look_around(coordinates_of_edge, cur_point, previous_point, existing_edge):

        coordinates_of_edge = coordinates_of_edge.tolist()
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
            if point[0] != cur_point[0] and point[1] != cur_point[1] and point not in existing_edge:
                return point, False

        for point in touching_points:
            # Due to segmentation faults, there are special cases -
            # then choose point further from the previous point
            if point[0] != previous_point[0] and point[1] != previous_point[1] \
                    and point not in existing_edge:
                return point, False

        return cur_point, True

    @staticmethod
    def _expand_edge(sorted_edge_axis):
        beg_ = np.ones(10) * sorted_edge_axis[0]
        end_ = np.ones(10) * sorted_edge_axis[-1]
        return np.concatenate((beg_, sorted_edge_axis, end_))

    @staticmethod
    def _make_grey_mask(mask):

        if np.any(mask.size != (256, 256)):
            print('segmented mask size: {}'.format(mask.shape))
            mask = cv2.resize(np.array(mask), (256, 256))
        if mask.shape == (256, 256, 3):
            mask_gray = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)

        # Remindex atrium
        mask_gray[mask_gray == 255] = 250
        return mask_gray

    @staticmethod
    def _pair_coordinates(edge):
        indices = np.where(edge == 255)
        y_pos = -indices[0]
        x_pos = indices[1]
        return np.array([(x, y) for x, y in zip(x_pos, y_pos)])

    def _walk_on_edge(self, coordinates_of_edge):
        """
        Since the ventricle is usually not convex, using radial coordinates can be misleading. A simple search
        deals with the proper order of the points and ensures a single-pixel edge.
        :param coordinates_of_edge: list of (x, y) coordinates of the edge on 2D plane
        :return: sorted list of coordinates of the edge
        """
        sorted_edge = list()
        edge_points = list()
        cur_point = tuple(max(coordinates_of_edge, key=lambda x: x[1]))
        prev_point = cur_point
        sorted_edge.append(cur_point)
        while 1:
            try:
                new_point, flag = self._look_around(coordinates_of_edge, cur_point, prev_point, sorted_edge)
            except TypeError:
                plt.scatter(sorted_edge, s=1)
                plt.xlim((0, 256))
                plt.ylim((-256, 0))
                self._save_failed_qc_image('Search for new point failed')

            if flag:
                edge_points.append(cur_point)
                if len(edge_points) == 2:
                    break
                sorted_edge.reverse()
                cur_point = sorted_edge[-1]
                prev_point = sorted_edge[-2]
            else:
                prev_point = cur_point
                cur_point = new_point
                sorted_edge.append(cur_point)

        basal_septal_edge = min(edge_points, key=lambda x: x[0])
        if basal_septal_edge != sorted_edge[0]:
            sorted_edge.reverse()

        return sorted_edge

    def _correct_indices(self):
        # To make sure that endocardium and not the cavity is captured, relevant indices are moved by 1
        row_diffs = np.diff(self.current_gray_mask, axis=1)
        row_diffs_right = np.where(row_diffs == self.mask_edge_value)
        row_diffs_left = np.where(row_diffs == 256 - self.mask_edge_value)
        col_diffs = np.diff(self.current_gray_mask, axis=0)
        col_diffs_down = np.where(col_diffs == self.mask_edge_value)
        col_diffs_up = np.where(col_diffs == 256 - self.mask_edge_value)

        if self.endo:  # index correction
            row_diffs_right = [row_diffs_right[0], row_diffs_right[1] + 1]
            col_diffs_down = [col_diffs_down[0] + 1, col_diffs_down[1]]
        else:
            row_diffs_left = [row_diffs_left[0], row_diffs_left[1] + 1]
            col_diffs_up = [col_diffs_up[0] + 1, col_diffs_up[1]]

        edge = list()
        edge.append(row_diffs_left)
        edge.append(row_diffs_right)
        edge.append(col_diffs_up)
        edge.append(col_diffs_down)


        return edge

    def _extract_edge_image(self):
        _lv_edge = np.zeros(self.current_gray_mask.shape, dtype=np.uint8)
        corrected_indices = self._correct_indices()
        for edge_pixels in corrected_indices:
            _lv_edge[edge_pixels] = 255

        return _lv_edge

    def _fit_border_through_pixels(self, new_resolution):

        se_x = np.array([x[0] for x in self.endo_sorted_edge])
        se_y = np.array([y[1] for y in self.endo_sorted_edge])
        fx = UnivariateSpline(range(len(se_x)), se_x, s=int(len(se_x)/6))
        fy = UnivariateSpline(range(len(se_y)), se_y, s=int(len(se_y)/10))
        ss = np.linspace(0, len(se_x), new_resolution)
        fitted_x = fx(ss)
        fitted_y = fy(ss)

        return [(x, y) for x, y in zip(fitted_x, fitted_y)]

        # plt.scatter(range(len(se_x) - 20), se_x[10:-10])
        # plt.plot(fitted_x, 'k', lw=4)
        # plt.show()

        # se_x = self._expand_edge(np.array([x[0] for x in self.endo_sorted_edge]))
        # se_y = self._expand_edge(np.array([y[1] for y in self.endo_sorted_edge]))
        # rbf_fx = Rbf(np.arange(len(se_x)), se_x, smooth=500, function='cubic')
        # rbf_fy = Rbf(np.arange(len(se_y)), se_y, smooth=500, function='cubic')
        # rbf_x = rbf_fx(ss)[10:-10]
        # rbf_y = rbf_fy(ss)[10:-10]
        # plt.plot(rbf_x, 'g', lw=4)
        # interp_se_x = self.smooth(se_x)[17:-17]
        # interp_se_y = self.smooth(se_y)[17:-17]
        # print('original: {}'.format(len(np.array([x[0] for x in self.endo_sorted_edge]))))
        # print('transformed: {}'.format(len(se_x)))
        # print('fitted: {}'.format(len(fitted_x)))
        # plt.plot(interp_se_x, 'r', lw=3)

        # print('len(fitted_x) {}'.format(len(fitted_x)))
        # print(len(fitted_y))
        # print('len(self.endo_sorted_edge): {}'.format(len(self.endo_sorted_edge)))
        # plt.plot(fitted_x, fitted_y)
        # plt.show()

    def _lv_edges(self, seg_mask):

        self.current_gray_mask = self._make_grey_mask(seg_mask)
        print(self.current_gray_mask.shape)
        self.mask_edge_value = 85 if self.endo else 170
        current_lv_edge = self._extract_edge_image()
        coord_lv = self._pair_coordinates(current_lv_edge)
        coord_lv_ordered = self._walk_on_edge(coord_lv)

        if len(coord_lv_ordered) < 100:
            self._save_failed_qc_image('Not enough edge points!', True)

        return coord_lv_ordered

    def lv_endo_edges(self):

        if self.segmentations_path is not None:
            for seg_file in self.seg_files:
                print(seg_file)
                seg_mask = cv2.imread(seg_file)
                self.endo = True
                self.endo_sorted_edge = self._lv_edges(seg_mask)
                self.endo = False
                self.epi_sorted_edge = self._lv_edges(seg_mask)
                #TODO: Pick the same y and see what happens
                #TODO: The smoothing should accept contour, not accept endo edge
                plt.imshow(self.current_gray_mask)
                xc = np.array([x[0] for x in self.endo_sorted_edge])
                yc = -np.array([y[1] for y in self.endo_sorted_edge])
                plt.plot(xc, yc, 'r')

                xc = np.array([x[0] for x in self.epi_sorted_edge])
                yc = -np.array([y[1] for y in self.epi_sorted_edge])
                plt.plot(xc, yc, 'b')
                plt.show()

                self._save_results(basename(seg_file)[:-4])

        elif self.segmentation_cycle is not None:

            print('length of the cycle: {}'.format(len(self.segmentation_cycle)))
            cycle_coords = []
            self.endo_sorted_edge = self._lv_edges(self.segmentation_cycle[0])
            prev_cont = self.endo_sorted_edge

            failed = 0
            for seg_img_i, seg_img in enumerate(self.segmentation_cycle):
                self.endo_sorted_edge = self._lv_edges(seg_img)
                self.img_index = seg_img_i
                if self._check_contour_quality(np.array(seg_img), prev_cont):
                    cycle_coords.append(self._fit_border_through_pixels(500))
                    prev_cont = self.endo_sorted_edge
                    # se_x = np.array([x[0] for x in self._fit_border_through_pixels(500)])
                    # se_y = np.array([y[1] for y in self._fit_border_through_pixels(500)])
                    # plt.imshow(seg_img)
                    # plt.scatter(se_x, -se_y)
                    # plt.show()
                else:
                    failed += 1

            if not failed / len(self.segmentation_cycle) > 0.35:
                self.all_cycle = cycle_coords
            else:
                self.all_cycle = None

    def _check_contour_quality(self, mask, prev_cont):

        values, counts = np.unique(mask, return_counts=True)  # returned array is sorted

        tmp_smooth = self._fit_border_through_pixels(len(self.endo_sorted_edge))
        # se_x = np.array([x[0] for x in self.endo_sorted_edge])
        # se_y = np.array([y[1] for y in self.endo_sorted_edge])
        # se_x2 = np.array([x[0] for x in prev_cont])
        # se_y2 = np.array([y[1] for y in prev_cont])
        # plt.scatter(se_x, -se_y)
        # plt.scatter(se_x2, -se_y2, marker='d')
        # plt.imshow(mask, cmap='gray')
        # print('a')
        # plt.show()
        # print('b')

        positions = np.where(mask == values[1])
        max_bp_y, max_bp_x = [np.max(p) for p in positions]
        min_bp_y, min_bp_x = [np.min(p) for p in positions]
        max_cont_y = np.max([-c[1] for c in tmp_smooth])
        min_cont_y = np.min([-c[1] for c in tmp_smooth])
        max_cont_x = np.max([c[0] for c in tmp_smooth])
        min_cont_x = np.min([c[0] for c in tmp_smooth])

        percent_prev_contour_diff = np.abs(len(self.endo_sorted_edge) - len(prev_cont)) / len(prev_cont)
        if percent_prev_contour_diff > 0.25:  # 25% of previous contour
            print(percent_prev_contour_diff)
            # self._save_failed_qc_image('percent_prev_cont {}'.format(percent_prev_contour_diff), mask)
            return False

        if np.linalg.norm(np.array(tmp_smooth[0]) - np.array(tmp_smooth[-1])) < 10:
            print(tmp_smooth)
            print(np.linalg.norm(np.array(tmp_smooth) - np.array(tmp_smooth)))
            # self._save_failed_qc_image('Contour covering entire bp', mask)
            return False

        if max_bp_y < max_cont_y - 5:
            # self._save_failed_qc_image('Contour over bp', mask)
            return False

        diff_max_y = np.abs(max_bp_y - max_cont_y)
        if diff_max_y > 10:
            # self._save_failed_qc_image('Diff_max_bp_y_max_cont_y {}'.format(diff_max_y), mask)
            return False

        diff_min_y = np.abs(min_bp_y - min_cont_y)
        if diff_min_y > 15:
            # self._save_failed_qc_image('Diff_min_bp_y_min_cont_y {}'.format(diff_min_y), mask)
            return False

        diff_max_x = np.abs(max_bp_x - max_cont_x)
        if diff_max_x > 15:
            # self._save_failed_qc_image('Diff_max_bp_x_max_cont_x {}'.format(diff_max_x), mask)
            return False

        diff_min_x = np.abs(min_bp_x - min_cont_x)
        if diff_min_x > 15:
            # self._save_failed_qc_image('Diff_min_bp_x_min_cont_x {}'.format(diff_min_x), mask)
            return False

        return True

    def _save_results(self, basename_file):
        out_dir = self._check_directory(os.path.join(self.output_path, 'Contour_tables'))
        np.savetxt(os.path.join(out_dir, basename_file + '.csv'), self.endo_sorted_edge)
        plt.imshow(self.endo_sorted_edge, cmap='gray')
        out_dir = self._check_directory(os.path.join(self.output_path, 'Edge_images'))
        plt.savefig(os.path.join(out_dir, basename_file + '.png'))
        plt.clf()
        self.endo_sorted_edge = np.array(self.endo_sorted_edge)
        plt.plot(self.endo_sorted_edge[:, 0], self.endo_sorted_edge[:, 1], 'k-')
        plt.xlim((0, 256))
        plt.ylim((-256, 0))
        plt.savefig(os.path.join(out_dir, basename_file + '_ordered.png'))
        plt.clf()

    def _save_failed_qc_image(self, plot_title, mask=False):
        if mask is not None:
            plt.imshow(self.current_gray_mask)
        plt.plot([x[0] for x in self.endo_sorted_edge], [-y[1] for y in self.endo_sorted_edge], 'r')
        plt.title(plot_title)
        case_dir = check_directory(os.path.join(self.output_path, 'failed_qc',
                                                self.s_sopid.replace('.', '_')))
        print(case_dir)
        shutil.rmtree('{}/*'.format(case_dir), ignore_errors=True)
        failed_dir = check_directory(os.path.join(self.output_path, 'failed_qc',
                                                  self.s_sopid.replace('.', '_'), str(self.cycle_index)))
        plt.savefig(os.path.join(failed_dir, '{}_{}_{}.png'.format(self.img_index, plot_title,
                                                                   self.cycle_index)))
        plt.close()


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Create contours to input into the curvature model')
    # parser.add_argument('-s', '--segmentations', help='Segmentation results from NTNU model', required=True)
    # parser.add_argument('-o', '--output_path', help='Directory where output will be stored', required=True)
    # args = parser.parse_args()

    cont = Contour(os.getcwd(), os.getcwd())
    cont.lv_endo_edges()
