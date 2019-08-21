import numpy as np
import glob
import os
import imageio
from skimage import measure
from PIL import Image
from shutil import move


class ProcessMRI:

    def __init__(self, path_to_png_images, output_path):
        self.path_to_png_images = path_to_png_images
        self.output_path = output_path

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

    def refine_images(self, background=0):

        for png_filename in glob.glob(os.path.join(self.path_to_png_images, '*.png')):
            # Read file and change everything that's not background into 1
            png = imageio.imread(png_filename)
            corr_png = np.copy(png)
            corr_png[corr_png != background] = 1
            # Create a mask to cover everything except the largest connected part that is not the background
            mask = measure.label(corr_png)
            unique, counts = np.unique(mask, return_counts=True)
            counts[unique == background] = 0
            largest_connected_label = unique[np.argmax(counts)]
            png[mask != largest_connected_label] = background
            # Save image without artifacts
            corrected_png = Image.fromarray(png)
            corrected_png.convert('L')
            corrected_png.save(os.path.join(self.output_path, os.path.basename(png_filename)))

    def move_low_quality_images(self):
        missing_septal_base_dir = self._check_directory(os.path.join(self.output_path, 'missing_basal_septum'))
        for png_filename in glob.glob(os.path.join(self.path_to_png_images, '*.png')):
            png = imageio.imread(png_filename)

            min_bpx, min_bpy, max_bpx, max_bpy = self._find_extreme_coordinates(png, 127)
            min_myox, min_myoy, max_myox, max_myoy = self._find_extreme_coordinates(png, 255)

            if min_bpx < min_myox:

                move(os.path.join(self.path_to_png_images, os.path.basename(png_filename)),
                     os.path.join(missing_septal_base_dir, os.path.basename(png_filename)))


if __name__ == '__main__':
    path_to_nifti_images = '/home/mat/Pictures/LAX_UKBB_corr'
    output_path = '/home/mat/Pictures/LAX_UKBB_corr'
    pr = ProcessMRI(path_to_nifti_images, output_path)
    pr.move_low_quality_images()
