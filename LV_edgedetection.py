import os
import glob
from ntpath import basename
import argparse
import numpy as np
from matplotlib import pyplot as plt
import cv2


class Contour:

    def __init__(self, segmentations_path, output_path):
        self.segmentations_path = segmentations_path
        self.output_path = self._check_directory(output_path)
        self.seg_files = glob.glob(os.path.join(self.segmentations_path, '*.png'))
        self.seg_files.sort()
        self.current_gray_mask = None
        self.sorted_edge = list()

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
                if [i, j] in coordinates_of_edge and\
                        np.all((i, j) != cur_point) and \
                        np.all((i, j) != previous_point):
                    touching_points.append((i, j))

        if len(touching_points) == 1:  # If only one touching point was found return
            return touching_points[0], False

        for point in touching_points:  # If more points were found, return diagonal one
            if point[0] != cur_point[0] and point[1] != cur_point[1] and point not in self.sorted_edge:
                return point, False

        for point in touching_points:  # Due to segmentation faults, there are special cases - then chose
            # point further from the previous point
            if point[0] != previous_point[0] and point[1] != previous_point[1] \
                    and point not in self.sorted_edge:
                return point, False

        # print('Edge point found')
        return cur_point, True

    def _walk_on_edge(self, coordinates_of_edge):
        """
        Since the ventricle is usually not convex, using radial coordinates can be misleading. A simple search
        deals with the proper order of the points and ensures a single-pixel edge.
        :param coordinates_of_edge: list of (x, y) coordinates of the edge on 2D plane
        :return: sorted list of coordinates of the edge
        """
        self.sorted_edge = list()
        edge_points = list()
        cur_point = tuple(min(coordinates_of_edge, key=lambda x: x[1]))
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

        return np.array(self.sorted_edge)

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

    def lv_endo_edges(self):

        for seg_file in self.seg_files:

            print(seg_file)
            seg_mask = cv2.imread(seg_file)
            # This resolution depends on the segmentation resolution. Perhaps could be better?
            seg_mask = cv2.resize(seg_mask, (256, 256))
            seg_mask_gray = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)

            self.current_gray_mask = seg_mask_gray
            current_lv_edge = self._extract_edge_image()
            coord_lv = self._pair_coordinates(current_lv_edge)
            _ = self._walk_on_edge(coord_lv)

            self._save_results(current_lv_edge, basename(seg_file)[:-4])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create contours to input into the curvature model')
    parser.add_argument('-s', '--segmentations', help='Segmentation results from NTNU model', required=True)
    parser.add_argument('-o', '--output_path', help='Directory where output will be stored', required=True)
    args = parser.parse_args()
    cont = Contour(args.segmentations, args.output_path)
    cont.lv_endo_edges()
