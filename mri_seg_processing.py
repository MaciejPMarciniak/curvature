import numpy as np
import glob
import os
import imageio
from skimage import measure
from PIL import Image
from shutil import move
import matplotlib.pyplot as plt


class ProcessMRI:

    def __init__(self, path_to_png_images, output_path, background=0):
        self.path_to_png_images = path_to_png_images
        self.output_path = output_path
        self.background = background

    @staticmethod
    def _check_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        return directory

    @staticmethod
    def _find_extreme_coordinates(mask, segment_value):
        positions = np.where(mask == segment_value)
        min_y, min_x = [np.min(p) for p in positions]
        max_y, max_x = [np.max(p) for p in positions]
        return min_x, min_y, max_x, max_y

    def set_path_to_png_images(self, new_path=''):
        self.path_to_png_images = new_path

    def set_output_path(self, new_path=''):
        self.output_path = new_path

    def clear_png(self, img_array):
        binary_img_array = np.copy(img_array)
        binary_img_array[binary_img_array != self.background] = 1
        mask = measure.label(binary_img_array)
        unique, counts = np.unique(mask, return_counts=True)
        counts[unique == self.background] = 0
        largest_connected_label = unique[np.argmax(counts)]
        img_array[mask != largest_connected_label] = self.background
        return img_array

    def create_border_on_lvbp(self, img_array):
        lvbp = np.where(img_array == 85)
        border = set()
        for pix, piy in zip(lvbp[0], lvbp[1]):
            if img_array[pix + 1, piy] == self.background:
                border.add((pix + 1, piy))
            if img_array[pix - 1, piy] == self.background:
                border.add((pix - 1, piy))
            if img_array[pix, piy + 1] == self.background:
                border.add((pix, piy + 1))
            if img_array[pix, piy - 1] == self.background:
                border.add((pix, piy - 1))

        for b_pi in border:
            img_array[b_pi] = 250
        return img_array

    def refine_images(self):

        for png_filename in glob.glob(os.path.join(self.path_to_png_images, '*.png')):
            png = imageio.imread(png_filename)

            # Quality check
            unique_masks = np.unique(png)
            if len(unique_masks) < 3:
                continue
            png = self.clear_png(png)  # Create a mask to cover everything except the largest connected segment
            png = self.create_border_on_lvbp(png)   # Add a layer of pixels at the base of LVbp

            # Save image without artifacts
            corrected_png = Image.fromarray(png)
            corrected_png.convert('L')
            corrected_png.save(os.path.join(self.output_path, os.path.basename(png_filename)))

    def move_low_quality_images(self):

        missing_septal_base_dir = self._check_directory(os.path.join(self.output_path, 'missing_basal_septum'))
        for png_filename in glob.glob(os.path.join(self.path_to_png_images, '*.png')):
            print(png_filename)
            png = imageio.imread(png_filename)

            min_bpx, min_bpy, max_bpx, max_bpy = self._find_extreme_coordinates(png, 85)
            min_myox, min_myoy, max_myox, max_myoy = self._find_extreme_coordinates(png, 170)

            if min_bpx < min_myox:

                move(os.path.join(self.path_to_png_images, os.path.basename(png_filename)),
                     os.path.join(missing_septal_base_dir, os.path.basename(png_filename)))


if __name__ == '__main__':
    path_to_nifti_images = '/home/mat/Pictures/LAX_UKBB'
    output_path = '/home/mat/Pictures/LAX_UKBB_corr'
    pr = ProcessMRI(path_to_nifti_images, output_path)
    pr.refine_images()
    # pr.set_path_to_png_images('/home/mat/Pictures/LAX_UKBB_corr')
    # pr.move_low_quality_images()
