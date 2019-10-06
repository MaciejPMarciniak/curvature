import os
import glob
from ntpath import basename
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
import shutil
from scipy.spatial.distance import cdist
from curvature import Curvature
from oct2py import Oct2Py


def check_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


class Contour:

    def __init__(self, segmentations_path, output_path, image_info_file='', segmentation_cycle=None, s_sopid=None,
                 cycle_index=None, dimensions=None):
        self.segmentations_path = segmentations_path
        self.output_path = check_directory(output_path)
        self.image_info_file = image_info_file
        self.segmentation_cycle = segmentation_cycle
        self.s_sopid = s_sopid
        self.cycle_index = cycle_index
        self.img_index = 0
        self.dimensions = dimensions
        if self.segmentations_path is not None:
            self.seg_files = glob.glob(os.path.join(self.segmentations_path, '*.png'))
            self.seg_files.sort()
        self.gray_mask = None
        self.all_cycle = None

        self.endo_sorted_edge = list()
        self.epi_sorted_edge = list()
        self.mask_values = {'LV_bp': 85, 'LV_myo': 170}
        self.is_lv_endo = True
        self.smoothing_resolution = 500
        self.distance_matrix = np.zeros(1)

    @staticmethod
    def _pair_coordinates(edge):
        return np.array([(x, y) for x, y in zip(edge[0], edge[1])])

    def _reindex_atrium(self):
        self.gray_mask[self.gray_mask == 255] = 250

    def _correct_indices(self):
        # To make sure that endocardium and not the cavity is captured, relevant indices are moved by 1
        row_diffs = np.diff(self.gray_mask, axis=1)
        row_diffs_right = list(np.where(row_diffs == self.edge_diff_value))
        row_diffs_left = list(np.where(row_diffs == 256 - self.edge_diff_value))
        col_diffs = np.diff(self.gray_mask, axis=0)
        col_diffs_down = list(np.where(col_diffs == self.edge_diff_value))
        col_diffs_up = list(np.where(col_diffs == 256 - self.edge_diff_value))

        # index correction
        row_diffs_left = [row_diffs_left[0], row_diffs_left[1] + 0.5]
        col_diffs_up = [col_diffs_up[0] + 0.5, col_diffs_up[1]]
        row_diffs_right = [row_diffs_right[0], row_diffs_right[1] + 0.5]
        col_diffs_down = [col_diffs_down[0] + 0.5, col_diffs_down[1]]

        # putting points together
        edge = list()
        edge_y = np.concatenate((row_diffs_right[0], row_diffs_left[0], col_diffs_down[0], col_diffs_up[0]))
        edge_x = np.concatenate((row_diffs_right[1], row_diffs_left[1], col_diffs_down[1], col_diffs_up[1]))
        edge.append(edge_x)
        edge.append(edge_y)

        return self._pair_coordinates(edge)

    def _find_closest_point(self, coordinates_of_edge, cur_point, existing_edge):

        # find closest point ids
        cur_point_id = coordinates_of_edge.tolist().index(cur_point)
        closest_point_id = np.where(self.distance_matrix[cur_point_id, :] <= 1.)[0]

        # retrieve points that were not picked before
        closest_points = [cpi for cpi in coordinates_of_edge[closest_point_id].tolist() if cpi not in existing_edge]

        # choose next point
        if len(closest_points) == 1:
            return closest_points[0], False

        # go left if more than one close point found
        if len(closest_points) > 1:
            closest_points.sort(key=lambda x: x[0])
            return closest_points[0], False

        # if no point found, flag the boundary
        return cur_point, True

    def _walk_on_edge(self, coordinates_of_edge):
        """
        Since the ventricle is usually not convex, using radial coordinates can be misleading. A simple search
        deals with the proper order of the points and ensures a single-pixel edge.
        :param coordinates_of_edge: list of (x, y) coordinates of the edge on 2D plane
        :return: sorted list of coordinates of the edge
        """
        sorted_edge = list()
        edge_points = list()

        self.distance_matrix = cdist(coordinates_of_edge, coordinates_of_edge, metric='euclidean')
        self.distance_matrix[self.distance_matrix == 0] = 100

        cur_point = list(min(coordinates_of_edge, key=lambda t: t[1]))
        sorted_edge.append(cur_point)
        while 1:
            try:
                new_point, flag = self._find_closest_point(coordinates_of_edge, cur_point, sorted_edge)
            except TypeError:
                plt.scatter(sorted_edge, s=1)
                plt.xlim((0, 256))
                plt.ylim((-256, 0))
                self._save_failed_qc_image('Search for new point failed')
                break

            if flag:
                edge_points.append(cur_point)
                if len(edge_points) == 2:
                    break
                sorted_edge.reverse()
                cur_point = sorted_edge[-1]
            else:
                cur_point = new_point
                sorted_edge.append(cur_point)

        basal_septal_edge = min(edge_points, key=lambda x: x[0])
        if np.all(basal_septal_edge != sorted_edge[0]):
            sorted_edge.reverse()

        return sorted_edge

    def _fit_border_through_pixels(self, edge=None):
        """
        Prerequisites to run matlab code from python:
        download & install Java Runtime Environment
        download & install Octave
        add all necessary paths to used files
        :param edge: if provided, given set of points is smoothed (implies that the points are sorted in some way).
        Otherwise one of the attributes is picked, based on the value of self.is_lv_endo:
        True -> lv_endo_sorted_edge
        False -> lv_epi_sorted_edge
        :return: list of points of the smooth contour, with resolution controlled by smoothing_resolution
        """
        # Add paths with relevant scripts and functions
        matlab_code_path = os.path.join('C:/', 'Code', 'computationalcardiacanatomy')
        pth = [x[0] for x in os.walk(os.path.join(matlab_code_path, 'Accesory', 'MeshHandling'))]
        pth.append(os.path.join(matlab_code_path, 'Personalization', 'CoreFunctions'))
        pth.append(os.path.join(matlab_code_path, 'Personalization', 'AccessoryFunctions'))
        pth.append(os.path.join(matlab_code_path, 'Scripts'))

        if edge is not None:
            border = edge
        elif self.is_lv_endo:
            border = self.endo_sorted_edge
        else:
            border = self.epi_sorted_edge
            self.smoothing_resolution = 2500

        print('Fitting')
        oc = Oct2Py()
        [oc.addpath(sub_pth) for sub_pth in pth]
        # Get smooth contour. 12 is the number of elements (why?)
        fit = oc.ScriptFitUltrasoundContour(border, 12, self.smoothing_resolution)
        print('Fitting complete')
        fitted_points = [(p[0], p[1]) for p in fit]

        return fitted_points, border

    def _retrieve_voxel_size(self, segmentation_file):

        try:
            df_image_info = pd.read_csv(self.image_info_file, index_col='id')
        except FileNotFoundError:
            exit('No image information file has been found\n'
                 'Provide the image information file to adjust the contour to the voxel size')

        image_id = os.path.basename(segmentation_file).split('.')[0]
        self.dimensions = df_image_info.loc[image_id][['voxel_size_width', 'voxel_size_height']].values

    def _scale_contours(self, segmentation_file):
        # Rescaling to metric units (millimeters)
        self._retrieve_voxel_size(segmentation_file)
        if self.is_lv_endo:
            self.endo_sorted_edge = [[x[0] * self.dimensions[0],
                                      x[1] * self.dimensions[1]] for x in self.endo_sorted_edge]
        else:
            self.epi_sorted_edge = [[x[0] * self.dimensions[0],
                                     x[1] * self.dimensions[1]] for x in self.epi_sorted_edge]

    def _lv_edges(self):

        self._reindex_atrium()
        self.edge_diff_value = self.mask_values['LV_bp'] if self.is_lv_endo else self.mask_values['LV_myo']
        lv_edge_points = self._correct_indices()
        lv_ordered_contour = self._walk_on_edge(lv_edge_points)

        if len(lv_ordered_contour) < 50:
            self._save_failed_qc_image('Not enough edge points!', True)

        return lv_ordered_contour

    # -----WallThicknessMeasurements------------------------------------------------------------------------------------
    def _calculate_bidirectional_local_distance_matrix(self):
        bidirectional_local_distance = {}
        cost_matrix = cdist(np.array(self.endo_sorted_edge), np.array(self.epi_sorted_edge))

        forward_min_distance = np.argmin(cost_matrix, axis=1)
        for p_ref, p_t in enumerate(forward_min_distance):
            bidirectional_local_distance[p_ref] = p_t

        backward_min_distance = np.argmin(cost_matrix, axis=0)
        for p_t, p_ref in enumerate(backward_min_distance):
            if cost_matrix[p_ref, p_t] > cost_matrix[p_ref, bidirectional_local_distance[p_ref]]:
                bidirectional_local_distance[p_ref] = p_t

        return bidirectional_local_distance

    def _calculate_wt(self):

        print('Calculating wall thickness')
        bld = self._calculate_bidirectional_local_distance_matrix()
        thickness = np.zeros(len(self.endo_sorted_edge))

        for point in range(len(thickness)):
            endo_point = self.endo_sorted_edge[point]
            epi_point = self.epi_sorted_edge[bld[point]]
            thickness[point] = np.linalg.norm(np.array(endo_point) - np.array(epi_point))

        basal_thicknesses = thickness[:83]
        mid_thicknesses = thickness[83:166]

        max_basal_thickness = np.max(basal_thicknesses)
        mean_mid_thickness = np.mean(mid_thicknesses)

        return max_basal_thickness, mean_mid_thickness
    # ---ENDWallThicknessMeasurements-----------------------------------------------------------------------------------

    # -----MainFunction-------------------------------------------------------------------------------------------------
    def lv_edges(self, calculate_wt):

        if self.segmentations_path is not None:
            for seg_file in self.seg_files:
                print(seg_file)
                self.gray_mask = imageio.imread(seg_file)
                self.is_lv_endo = True
                self.endo_sorted_edge = self._lv_edges()
                # self._scale_contours(seg_file)
                self.plot_mask_with_contour(None, self.endo_sorted_edge)
                self.endo_sorted_edge, orig = self._fit_border_through_pixels()
                self.plot_mask_with_contour(self.endo_sorted_edge)
                # self._save_results(basename(seg_file)[:-4])

                if calculate_wt:
                    # self.is_lv_endo = False
                    # self.epi_sorted_edge = self._lv_edges()
                    # self._scale_contours(seg_file)
                    # self.epi_sorted_edge, orig = self._fit_border_through_pixels()
                    # self.plot_mask_with_contour(self.endo_sorted_edge, self.epi_sorted_edge)
                    self.plot_mask_with_contour(self.endo_sorted_edge, orig)
                    bwt, mwt = self._calculate_wt()

                    self.plot_wt()
                    print('Basal maximum wt: {}, mid mean wt: {}'.format(bwt, mwt))

        elif self.segmentation_cycle is not None:

            print('length of the cycle: {}'.format(len(self.segmentation_cycle)))
            cycle_coords = []
            self.endo_sorted_edge = self._lv_edges()
            prev_cont = self.endo_sorted_edge

            failed = 0
            for seg_img_i, seg_img in enumerate(self.segmentation_cycle):
                self.endo_sorted_edge = self._lv_edges()
                self.img_index = seg_img_i

                if self._check_contour_quality(np.array(seg_img), prev_cont):
                    cycle_coords.append(self._fit_border_through_pixels())
                    prev_cont = self.endo_sorted_edge
                else:
                    failed += 1

            if not failed / len(self.segmentation_cycle) > 0.35:
                self.all_cycle = cycle_coords
            else:
                self.all_cycle = None
    # ---ENDMainFunction------------------------------------------------------------------------------------------------

    # -----Saving-------------------------------------------------------------------------------------------------------
    def _save_results(self, basename_file):
        out_dir = check_directory(os.path.join(self.output_path, 'Contour_tables'))
        np.savetxt(os.path.join(out_dir, basename_file + '.csv'), self.endo_sorted_edge)

        out_dir = check_directory(os.path.join(self.output_path, 'Edge_images'))
        plt.imshow(self.gray_mask)
        if self.endo_sorted_edge:
            plt.plot([x[0] for x in self.endo_sorted_edge], [y[1] for y in self.endo_sorted_edge], 'r--')
        if self.epi_sorted_edge:
            plt.plot([x[0] for x in self.endo_sorted_edge], [y[1] for y in self.endo_sorted_edge], 'b--')
        plt.savefig(os.path.join(out_dir, basename_file + '_ordered.png'))
        plt.clf()
    # ---ENDSaving------------------------------------------------------------------------------------------------------

    # -----Plotting-----------------------------------------------------------------------------------------------------
    def plot_mask_with_contour(self, contour1=None, contour2=None):

        plt.imshow(self.gray_mask, cmap='gray')
        if contour1 is not None:
            xc = np.array([x[0] for x in contour1])
            yc = np.array([y[1] for y in contour1])
            plt.plot(xc, yc, 'r.-')

        if contour2 is not None:
            xc = np.array([x[0] for x in contour2])
            yc = np.array([y[1] for y in contour2])
            plt.plot(xc, yc, 'b.-')
        plt.show()
        plt.clf()

    def plot_wt(self):

        bld = self._calculate_bidirectional_local_distance_matrix()
        plt.imshow(self.gray_mask, cmap='gray')
        for key in bld:
            xs = self.endo_sorted_edge[key][0], self.epi_sorted_edge[bld[key]][0]
            ys = self.endo_sorted_edge[key][1], self.epi_sorted_edge[bld[key]][1]
            plt.plot(xs, ys, 'y--')

        plt.plot([x[0] for x in self.epi_sorted_edge], [x[1] for x in self.epi_sorted_edge], 'b.')
        plt.plot([x[0] for x in self.endo_sorted_edge], [x[1] for x in self.endo_sorted_edge], 'r.')
        plt.show()
        plt.clf()
    # ---END Plotting---------------------------------------------------------------------------------------------------

    # -----QualityChecks------------------------------------------------------------------------------------------------
    def _check_contour_quality(self, mask, prev_cont):

        values, counts = np.unique(mask, return_counts=True)  # returned array is sorted

        tmp_smooth = self._fit_border_through_pixels(True, len(self.endo_sorted_edge))
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
        max_bp_y *= self.dimensions[1]  # scaling for compatibility with contours
        max_bp_x *= self.dimensions[0]
        min_bp_y, min_bp_x = [np.min(p) for p in positions]
        min_bp_y *= self.dimensions[1]
        min_bp_x *= self.dimensions[0]
        max_cont_y = np.max([-c[1] for c in tmp_smooth])
        min_cont_y = np.min([-c[1] for c in tmp_smooth])
        max_cont_x = np.max([c[0] for c in tmp_smooth])
        min_cont_x = np.min([c[0] for c in tmp_smooth])
        percent_prev_contour_diff = np.abs(len(self.endo_sorted_edge) - len(prev_cont)) / len(prev_cont)
        if percent_prev_contour_diff > 0.25:  # 25% of previous contour
            print('percent_prev_contour_diff: {}'.format(percent_prev_contour_diff))
            # self._save_failed_qc_image('percent_prev_cont {}'.format(percent_prev_contour_diff), mask)
            return False

        if np.linalg.norm(np.array(tmp_smooth[0]) - np.array(tmp_smooth[-1])) < 10:
            print('tmp_smooth {}'.format(tmp_smooth))
            print(np.linalg.norm(np.array(tmp_smooth) - np.array(tmp_smooth)))
            # self._save_failed_qc_image('Contour covering entire bp', mask)
            return False

        if max_bp_y < max_cont_y - 5:
            print('Contour over bp')
            # self._save_failed_qc_image('Contour over bp', mask)
            return False

        diff_max_y = np.abs(max_bp_y - max_cont_y)
        if diff_max_y > 10:
            print('Diff_max_bp_y_max_cont_y {}'.format(diff_max_y))
            # self._save_failed_qc_image('Diff_max_bp_y_max_cont_y {}'.format(diff_max_y), mask)
            return False

        diff_min_y = np.abs(min_bp_y - min_cont_y)
        if diff_min_y > 15:
            print('Diff_min_bp_y_min_cont_y {}'.format(diff_min_y))
            # self._save_failed_qc_image('Diff_min_bp_y_min_cont_y {}'.format(diff_min_y), mask)
            return False

        diff_max_x = np.abs(max_bp_x - max_cont_x)
        if diff_max_x > 15:
            print('Diff_max_bp_x_max_cont_x {}'.format(diff_max_x))
            # self._save_failed_qc_image('Diff_max_bp_x_max_cont_x {}'.format(diff_max_x), mask)
            return False

        diff_min_x = np.abs(min_bp_x - min_cont_x)
        if diff_min_x > 15:
            print('Diff_min_bp_x_min_cont_x {}'.format(diff_min_x))
            # self._save_failed_qc_image('Diff_min_bp_x_min_cont_x {}'.format(diff_min_x), mask)
            return False

        return True

    def _save_failed_qc_image(self, plot_title, mask=False):
        if mask is not None:
            plt.imshow(self.gray_mask)
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
# ---END QualityChecks--------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Create contours to input into the curvature model')
    # parser.add_argument('-s', '--segmentations', help='Segmentation results from NTNU model', required=True)
    # parser.add_argument('-o', '--output_path', help='Directory where output will be stored', required=True)
    # args = parser.parse_args()

    # LINUX
    # segmentations_path = '/home/mat/Pictures/LAX_UKBB_corr'
    # output_path = '/home/mat/Pictures/LAX_UKBB_corr/contours'
    # WINDOWS
    segmentations_path = 'C:\Data\ProjectCurvature\LAX_UKBB\corrected'
    output_path = 'C:\Data\ProjectCurvature\LAX_UKBB\corrected\contours'
    image_info_file = 'C:\Data\ProjectCurvature\LAX_UKBB\Images_info\Image_details.csv'

    # cont = Contour(os.getcwd(), os.getcwd(), dimensions=(1, 1))
    cont = Contour(segmentations_path, output_path, image_info_file=image_info_file)
    cont.lv_edges(calculate_wt=True)
