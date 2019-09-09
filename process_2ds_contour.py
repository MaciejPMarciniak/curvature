import os
import numpy as np
from read_dicom import _check_directory
from shutil import move
import glob
import pandas as pd

path_to_contours = r'D:\2DS_output'
output_path = r'D:\2DS_output\parsed'


def organize_contours_in_folders(_path):
    for dirpath, dirnames, filenames in os.walk(_path):
        for n, fname in enumerate(filenames):
            print(fname)
            if os.path.splitext(fname)[-1] == '.txt':
                dicom_file_base = fname[-20:-12]
                dicom_dir = _check_directory(os.path.join(_path, dicom_file_base))
                print(dicom_dir)
                move(os.path.join(dirpath, fname), dicom_dir)


def remove_atrial_strain_data(_path):
    contour_dirs = os.listdir(_path)

    for contour_folder in contour_dirs:
        print(contour_folder)
        try:
            data_0y = np.genfromtxt(glob.glob(os.path.join(_path, contour_folder, '*_0y*'))[0])
            if (np.inf in data_0y) or (np.max(data_0y) > data_0y[0]):
                move(os.path.join(_path, contour_folder), os.path.join(_path, '_irrelevant'))
        except IndexError:
            exit('Moving strain completed')


def parse_contour_data(_path):
    contour_dirs = os.listdir(_path)

    for contour_folder in contour_dirs:
        contour_files = os.listdir(os.path.join(_path, contour_folder))
        contour_files = [f for f in contour_files if os.path.splitext(f)[-1] == '.txt']
        dict_contour = {}
        print(contour_folder)
        for contour_file in contour_files:
            ref = contour_file.split('_')[1]
            dict_contour[ref] = np.genfromtxt(os.path.join(_path, contour_folder, contour_file)) * 1000

        df_contour = pd.DataFrame(dict_contour)
        df_cols = df_contour.columns.values
        df_cols = sorted(df_cols, key=lambda x: x[:-1].zfill(3))
        df_contour = df_contour[df_cols]
        df_contour.to_csv(os.path.join(_path, contour_folder, contour_folder + '.csv'))


# organize_contours_in_folders(path_to_contours)
# remove_atrial_strain_data(path_to_contours)
parse_contour_data(path_to_contours)

