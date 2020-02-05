import os
import glob
# import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
from scipy.spatial.distance import cdist
from scipy.interpolate.rbf import Rbf
from curvature import GradientCurvature
from datetime import datetime
from PIL import Image


def check_directory(directory):
    # Check if directory exists; if not create it
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


class Contour:

    def __init__(self, segmentations_path, output_path=None, image_info_file=None,  pixel_size=None, scale=False,
                 smoothing_resulution=500, plot_smoothing_results=False):
        """
        A class to calculate the curvature and wall thickness of the left ventricle based on the NTNU segmentation model
        :param segmentations_path: folder with segmentation images (with .png extension)
        :param output_path: results will be stored in this folder
        :param image_info_file: a csv file, containing information about the resolution of the original image. It should
        contain three columns - 'id', with the name of the segmentation file (e.g. segmentation123.png),
        and 'voxel_size_width' and 'voxel_size_height', with values that are used to resize the calculated markers to
        a proper scale.
        :param pixel_size: (tuple) if scalling factors are the same for all images in the segmentations_path, provide a
        tuple with width and height of the pixels (in this order)
        :param scale: (bool) whether or not to scale the markers
        :param smoothing_resulution: (int) the number of points in the smooth version of the endocardium. The epicardium
        will become 5 times as long, by default (for more accurate wall thickness measurement)
        :param plot_smoothing_results: (bool) whether or not to plot the results of smoothing
        """

        self.segmentations_path = segmentations_path
        if output_path is not None:
            self.output_path = check_directory(output_path)
        else:
            self.output_path = check_directory(os.path.join(segmentations_path, 'output'))

        if pixel_size is not None:
            self.pixel_size = pixel_size  # either provided explicitly
        else:
            self.pixel_size = (1, 1)

        self.image_info_file = image_info_file
        if self.pixel_size == (1, 1) and self.image_info_file is None:
            print('Warning! No dimensions of the mask files provided. Set to (1, 1)')

        if self.segmentations_path is not None:
            self.seg_files = glob.glob(os.path.join(self.segmentations_path, '*.png'))  # list of mask files
            self.seg_files.sort()

        self.scale = scale
        self.plot_smoothing_results = plot_smoothing_results

        self.img_index = 0
        self.gray_mask = None
        self.endo_sorted_edge = list()
        self.epi_sorted_edge = list()
        self.mask_values = {'LV_bp': 85, 'LV_myo': 170}  # in NTNU model, these were the default values
        self.is_endo = True
        self.smoothing_resolution = smoothing_resulution
        self.distance_matrix = np.zeros(1)
        self.wall_thickness = None
        self.wall_thickness_markers = None
        self.curvature = None
        self.endo_curvature_markers = None
        self.epi_curvature_markers = None

    @staticmethod
    def _pair_coordinates(edge):
        return np.array([(x, y) for x, y in zip(edge[0], edge[1])])

    def _reindex_atrium(self):
        self.gray_mask[self.gray_mask == 255] = 250  # Changed to make computation easier

    # -----CreatingBorderContours---------------------------------------------------------------------------------------
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
                plt.scatter(coordinates_of_edge)
                plt.xlim((0, 256))
                plt.ylim((-256, 0))
                plt.title('Border generation failed')
                plt.show()
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

        left_edge = min(edge_points, key=lambda x: x[0])
        if np.all(left_edge != sorted_edge[0]):
            sorted_edge.reverse()

        return sorted_edge

    def _fit_border_through_pixels(self, edge=None):
        """
        :param edge: if provided, given set of points is smoothed (implies that the points are sorted in some way).
        Otherwise one of the attributes is picked, based on the value of self.is_endo:
        True -> lv_endo_sorted_edge
        False -> lv_epi_sorted_edge
        :return: list of points of the smooth contour, with resolution controlled by smoothing_resolution
        """
        _smoothing_resolution = self.smoothing_resolution
        if edge is not None:
            border = edge
        elif self.is_endo:
            border = self.endo_sorted_edge
        else:
            border = self.epi_sorted_edge
            _smoothing_resolution *= 5

        print('Interpolating')

        x_orig = np.array([x[0] for x in border])
        y_orig = np.array([y[1] for y in border])

        positions = np.arange(len(border))  # strictly monotonic, number of points in single trace
        rbf_x = Rbf(positions, x_orig, smooth=np.power(10, 5), function='quintic') # Heavy smoothing due to pixel effect
        rbf_y = Rbf(positions, y_orig, smooth=np.power(10, 5), function='quintic') # Heavy smoothing due to pixel effect

        # Interpolate based on the RBF model
        interpolation_target_n = np.linspace(0, len(border) - 1, _smoothing_resolution)
        x_interpolated = rbf_x(interpolation_target_n)
        y_interpolated = rbf_y(interpolation_target_n)

        if self.plot_smoothing_results:
            # Plot if you want to see the results
            plt.plot(x_orig, y_orig, '.-')
            plt.plot(x_interpolated[::2], y_interpolated[1::2], 'r')
            plt.show()

        # Return interpolated trace
        fitted_points = [(point_x, point_y) for point_x, point_y in zip(x_interpolated, y_interpolated)]

        return fitted_points, border

    def _lv_edge(self):
        self._reindex_atrium()
        self.edge_diff_value = self.mask_values['LV_bp'] if self.is_endo else self.mask_values['LV_myo']
        lv_edge_points = self._correct_indices()
        lv_ordered_contour = self._walk_on_edge(lv_edge_points)

        return lv_ordered_contour
    # -----END-CreatingBorderContours-----------------------------------------------------------------------------------

    # -----Scaling------------------------------------------------------------------------------------------------------
    def _retrieve_voxel_size(self, segmentation_file):
        try:
            df_image_info = pd.read_csv(self.image_info_file, index_col='id')
            image_id = os.path.basename(segmentation_file)
            self.pixel_size = df_image_info.loc[image_id][['voxel_size_width', 'voxel_size_height']].values
        except FileNotFoundError:
            exit('No image information file has been found\n'
                 'Provide the image information file to adjust the contour to the voxel size')

    def _scale_contours(self, segmentation_file):
        if self.image_info_file != '':
            self._retrieve_voxel_size(segmentation_file)  # update dimensions

        if self.is_endo:
            self.endo_sorted_edge = [[x[0] * self.pixel_size[0],
                                      x[1] * self.pixel_size[1]] for x in self.endo_sorted_edge]
        else:
            self.epi_sorted_edge = [[x[0] * self.pixel_size[0],
                                     x[1] * self.pixel_size[1]] for x in self.epi_sorted_edge]
    # ---END-Scaling----------------------------------------------------------------------------------------------------

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

        return thickness

    def _get_wt_markers(self):
        thickness_segments = {
            'basal_thickness_1': self.wall_thickness[:int(self.smoothing_resolution / 6)],
            'mid_thickness_1': self.wall_thickness[
                               int(self.smoothing_resolution / 6):int(2*self.smoothing_resolution / 6)],
            'apical_thickness_1': self.wall_thickness[
                                  int(2*self.smoothing_resolution / 6):int(3*self.smoothing_resolution / 6)],
            'apical_thickness_2': self.wall_thickness[
                                  int(3*self.smoothing_resolution / 6):int(4*self.smoothing_resolution / 6)],
            'mid_thickness_2': self.wall_thickness[
                               int(4*self.smoothing_resolution / 6):int(5*self.smoothing_resolution / 6)],
            'basal_thickness_2': self.wall_thickness[int(5*self.smoothing_resolution / 6):],
        }
        thickness_markers = {}
        for key, value in thickness_segments.items():
            thickness_markers[key + '_mean'] = np.mean(value)
            thickness_markers[key + '_min'] = np.min(value)
            thickness_markers[key + '_max'] = np.max(value)

        return thickness_markers
    # ---END-WallThicknessMeasurements----------------------------------------------------------------------------------

    # -----CurvatureMeasurements----------------------------------------------------------------------------------------
    def _calculate_curvature(self):
        contour = self.endo_sorted_edge if self.is_endo else self.epi_sorted_edge
        curv = GradientCurvature(contour)
        return curv.calculate_curvature()

    def _get_curvature_markers(self):
        curvature_segments = {
            'basal_curvature_1': self.curvature[:int(self.smoothing_resolution / 6)],
            'mid_curvature_1': self.curvature[int(self.smoothing_resolution / 6):int(2*self.smoothing_resolution / 6)],
            'apical_curvature_1': self.curvature[
                                  int(2 * self.smoothing_resolution / 6):int(3 * self.smoothing_resolution / 6)],
            'apical_curvature_2': self.curvature[
                                  int(3 * self.smoothing_resolution / 6):int(4 * self.smoothing_resolution / 6)],
            'mid_curvature_2': self.curvature[
                               int(4 * self.smoothing_resolution / 6):int(5 * self.smoothing_resolution / 6)],
            'basal_curvature_2': self.curvature[int(5 * self.smoothing_resolution / 6):],
        }
        wall = 'endo' if self.is_endo else 'epi'
        curvature_markers = {}
        for key, value in curvature_segments.items():
            curvature_markers[key + '_mean_' + wall] = np.mean(value)
            curvature_markers[key + '_min_' + wall] = np.min(value)
            curvature_markers[key + '_max_' + wall] = np.max(value)

        return curvature_markers
    # ---END-CurvatureMeasurements--------------------------------------------------------------------------------------

    # -----ExecFunctions------------------------------------------------------------------------------------------------
    def _lv_edges(self, seg_file):

        print('Case file: {}'.format(os.path.basename(seg_file)))
        self.gray_mask = imageio.imread(seg_file)
        self.is_endo = True
        self.endo_sorted_edge = self._lv_edge()
        if self.scale:
            self._scale_contours(seg_file)
        self.endo_sorted_edge, orig = self._fit_border_through_pixels()
        self.is_endo = False
        self.epi_sorted_edge = self._lv_edge()
        if self.scale:
            self._scale_contours(seg_file)
        self.epi_sorted_edge, orig = self._fit_border_through_pixels()
        if self.plot_smoothing_results:
            self.plot_mask_with_contour(self.endo_sorted_edge, self.epi_sorted_edge)

    def _lv_markers(self, seg_file):
        self.wall_thickness = self._calculate_wt()
        self.wall_thickness_markers = self._get_wt_markers()

        if self.plot_smoothing_results:
            self.plot_wt()

        self.is_endo = True
        self.curvature = self._calculate_curvature()
        self.endo_curvature_markers = self._get_curvature_markers()
        self.is_endo = False
        self.curvature = self._calculate_curvature()
        self.epi_curvature_markers = self._get_curvature_markers()

        all_markers = {'caseID': os.path.basename(seg_file)}
        for markers in (self.wall_thickness_markers, self.endo_curvature_markers, self.epi_curvature_markers):
            all_markers.update(markers)

        return all_markers
    # ---END-ExecFunctions----------------------------------------------------------------------------------------------

    # -----Saving-------------------------------------------------------------------------------------------------------
    def save_all_results(self, save_markers=True, save_contours=False, save_wt=False, save_curvature=False,
                         save_images=False):
        markers_cohort = []
        for seg_file in self.seg_files:
            filename = os.path.basename(seg_file).split('.')[0]
            self._lv_edges(seg_file)
            if save_contours:
                out_dir = check_directory(os.path.join(self.output_path, 'Contours'))
                np.savetxt(os.path.join(out_dir, filename + '_endocardium.csv'), self.endo_sorted_edge)
                np.savetxt(os.path.join(out_dir, filename + '_epicardium.csv'), self.epi_sorted_edge)
            markers_cohort.append(self._lv_markers(seg_file))
            if save_images:
                out_dir = check_directory(os.path.join(self.output_path, 'Pictures'))
                self.plot_mask_with_contour(self.endo_sorted_edge, self.epi_sorted_edge,
                                            save_file=os.path.join(out_dir, filename + '_contours.png'))
                self.plot_wt(save_file=os.path.join(out_dir, filename + '_wall_thickness.png'))
            if save_wt:
                out_dir = check_directory(os.path.join(self.output_path, 'Wall_thickness'))
                np.savetxt(os.path.join(out_dir, filename + '_wall_thickness.csv'), self.wall_thickness)
            if save_curvature:
                out_dir = check_directory(os.path.join(self.output_path, 'Curvature'))
                np.savetxt(os.path.join(out_dir, filename + '_curvature.csv'), self.curvature)
            if save_markers:
                out_dir = check_directory(os.path.join(self.output_path, 'Markers'))
                df_cohort = pd.DataFrame(markers_cohort)
                date_id = datetime.now().isoformat(timespec='minutes').replace('T', '_').replace(':', '-')
                df_cohort.to_csv(os.path.join(out_dir, 'Cohort_markers_{}.csv'.format(date_id)))

    # ---END-Saving-----------------------------------------------------------------------------------------------------

    # -----Plotting-----------------------------------------------------------------------------------------------------
    def _scale_mask(self):
        pil_mask = Image.fromarray(self.gray_mask)
        new_size = [int(a*b) for a, b in zip(self.gray_mask.shape, self.pixel_size)]
        pil_mask = pil_mask.resize(new_size, Image.BILINEAR)
        return np.array(pil_mask)

    def plot_mask_with_contour(self, contour1=None, contour2=None, save_file=''):

        scaled_greyscale = self._scale_mask()
        plt.imshow(scaled_greyscale, cmap='gray')
        if contour1 is not None:
            xc = np.array([x[0] for x in contour1])
            yc = np.array([y[1] for y in contour1])
            plt.plot(xc, yc, 'r.-')

        if contour2 is not None:
            xc = np.array([x[0] for x in contour2])
            yc = np.array([y[1] for y in contour2])
            plt.plot(xc, yc, 'b.-')

        if save_file != '':
            plt.savefig(save_file)
        else:
            plt.show()
        plt.clf()

    def plot_wt(self, save_file=''):

        bld = self._calculate_bidirectional_local_distance_matrix()
        scaled_greyscale = self._scale_mask()
        plt.imshow(scaled_greyscale, cmap='gray')
        for key in bld:
            xs = self.endo_sorted_edge[key][0], self.epi_sorted_edge[bld[key]][0]
            ys = self.endo_sorted_edge[key][1], self.epi_sorted_edge[bld[key]][1]
            plt.plot(xs, ys, 'y--')

        plt.plot([x[0] for x in self.epi_sorted_edge], [x[1] for x in self.epi_sorted_edge], 'b.')
        plt.plot([x[0] for x in self.endo_sorted_edge], [x[1] for x in self.endo_sorted_edge], 'r.')

        if save_file != '':
            plt.savefig(save_file)
        else:
            plt.show()
        plt.clf()
    # ---END-Plotting---------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Create contours to input into the curvature model')
    # parser.add_argument('-s', '--segmentations', help='Segmentation results from NTNU model', required=True)
    # parser.add_argument('-o', '--output_path', help='Directory where output will be stored', required=True)
    # args = parser.parse_args()

    segmentations_path = 'C:\Code\curvature\model'
    output_path = segmentations_path

    cont = Contour(segmentations_path, output_path, image_info_file='')
    cont.save_all_results(save_markers=True, save_contours=False, save_wt=False, save_curvature=False,
                          save_images=False)
