import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import pandas as pd
from skimage import measure
import imageio

class ConvertNIfTI2PNG:

    def __init__(self, path_to_images, output_path, nifti_filename):
        self.path_to_images = path_to_images
        self.output_path = output_path
        self.nifti_filename = nifti_filename
        self.nifti = nib.load(os.path.join(self.path_to_images, self.nifti_filename))

    def print_nifti_info(self):
        print('shape: {}'.format(self.nifti.shape))
        print('data affine shape: {}'.format(self.nifti.affine.shape))
        print('data type: {}'.format(self.nifti.get_data_dtype()))
        hdr = self.nifti.header
        print('header:\n{}'.format(hdr))
        print('xyzt units: {}'.format(hdr.get_xyzt_units()))
        print('quaterns: {}'.format(hdr.get_qform_quaternion()))

    def show_nifti_image(self):
        n_image = np.squeeze(self.nifti.get_fdata())
        print('file name: {}'.format(self.nifti_filename))
        print('image shape: {}'.format(n_image.shape))
        plt.imshow(n_image[:, :], cmap='gray')
        plt.show()

    def save_nifti_image_as_png(self, output_path=None):
        if output_path is not None:
            self.output_path = output_path

        quaterns = self.nifti.header.get_qform_quaternion()
        n_image = np.squeeze(self.nifti.get_fdata())
        n_image[n_image > 2] = 0
        if np.abs(quaterns[1]) < np.abs(quaterns[3]):  # transpose if necessary
            n_image = np.transpose(n_image)
        n_image = n_image * 85
        png_image = Image.fromarray(n_image)
        png_image = png_image.convert("L")
        png_image.save(os.path.join(self.output_path, os.path.basename(self.nifti_filename.split('.')[0] + '.png')))
# -----END Class ConvertNIfTI2PNG---------------------------------------------------------------------------------------


def save_nifti_images_info(path_to_nifti_images='', output_path='', image_info_file='Image_details.csv'):

    nifti_images_list = []
    for n_file in glob.glob(os.path.join(path_to_nifti_images, '*')):
        print(n_file)
        nif = ConvertNIfTI2PNG(path_to_nifti_images, output_path, n_file)
        nif_info = dict()
        nif_info['id'] = os.path.basename(n_file).split('.')[0]
        nif_info['voxel_size_height'] = nif.nifti.header['pixdim'][1]
        nif_info['voxel_size_width'] = nif.nifti.header['pixdim'][2]
        nif_info['dim_height'] = nif.nifti.header['dim'][1]
        nif_info['dim_width'] = nif.nifti.header['dim'][2]

        quaterns = nif.nifti.header.get_qform_quaternion()
        # If the image was transposed, the dimensions and voxel sizes should be transposed accordingly:
        if np.abs(quaterns[1]) < np.abs(quaterns[3]):
            nif_info['voxel_size_height'], nif_info['voxel_size_width'] = \
                nif_info['voxel_size_width'], nif_info['voxel_size_height']
            nif_info['dim_height'], nif_info['dim_width'] = nif_info['dim_width'], nif_info['dim_height']
        nif_info['units'] = nif.nifti.header.get_xyzt_units()[0]
        nifti_images_list.append(nif_info)

    df_nifti = pd.DataFrame(nifti_images_list)
    df_nifti = df_nifti.set_index('id')
    df_nifti.to_csv(os.path.join(output_path, image_info_file))
    return df_nifti


# ----------------------------------------------------------------------------------------------------------------------


path_to_nifti_images = '/media/mat/D6B7-122E/LAX_UKBB/LAX_UKBB'
output_path = '/home/mat/Pictures/LAX_UKBB'

# ---Save image info
# save_nifti_images_info(path_to_nifti_images, output_path)

# ---Convert files
for n_file in glob.glob(os.path.join(path_to_nifti_images, '*')):

    ven = ConvertNIfTI2PNG(path_to_nifti_images, output_path, n_file)
    # ven.print_nifti_info()
    # ven.show_nifti_image()
    ven.save_nifti_image_as_png()


