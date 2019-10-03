import pydicom
import os
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
from shutil import move


def _check_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


# ---Copy2DStrainFiles--------------------------------------------------------------------------------------------------
def copy_2ds_sequences(path_to_dicom_folders):

    two_ds_dir = _check_directory(os.path.join(os.path.split(path_to_dicom_folders)[0], '2DStrain'))

    i = 0
    patient_ids = []
    patient_files = []

    for dirpath, dirnames, filenames in os.walk(path_to_dicom_folders):
        pid = True

        for dicom_filename in filenames:
            ds = pydicom.read_file(os.path.join(dirpath, dicom_filename))
            if pid:
                i += 1
                print(i)
                print('Case: {}'.format(ds.PatientID))
                print('Filename: {}'.format(os.path.split(dirpath)[-1]))
                patient_ids.append(ds.PatientID)
                patient_files.append(dirpath)
                pid = False

            if ds.ImageType[6] == 'GEMS2DSTRAIN':
                print(dicom_filename)
                two_ds_fol = _check_directory(os.path.join(two_ds_dir, ds.PatientID))
                shutil.copy(os.path.join(dirpath, dicom_filename),
                            os.path.join(two_ds_fol, dicom_filename+'_2DS'))

    with open(os.path.join(os.path.split(path_to_dicom_folders)[0], 'Patient_IDs.csv'), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(patient_ids)
        wr.writerow(patient_files)


def remove_modified_dicoms(path_to_dicom_folders):
    for (dirpath, dirnames, filenames) in os.walk(path_to_dicom_folders):
        for filename in filenames:
            if filename.endswith('_modified.dcm'):
                # os.remove(os.path.join(dirpath, filename))
                pass


def read_dicom(path_to_dicom_folders):
    for (dirpath, dirnames, filenames) in os.walk(path_to_dicom_folders):
        print(filenames)
        for dicom_filename in filenames:
            if dicom_filename == 'J9GD69G0':
                ds = pydicom.read_file(os.path.join(dirpath, dicom_filename))
                pa = ds.pixel_array
                plt.imshow(pa[400:650, 700:])
                plt.show()
                # plt.savefig('pixel_array.png')

                print(ds)


def find(name, path):
    for root, dirs, files in os.walk(path):
        for dicoms in files:
            if name in dicoms:
                return os.path.join(root, name)


def move_atrial_strain_files(path_to_dicom_folders, roi_folder_path):
    roi_files = os.listdir(roi_folder_path)

    for roi_file in roi_files:
        try:
            roi_data = np.genfromtxt(os.path.join(roi_folder_path, roi_file), delimiter=',')
            if roi_data[int(roi_data.shape[0]/2), 0] > np.mean((roi_data[0, 0], roi_data[-1, 0])):
                # print(roi_file)
                dicom_filename = roi_file.rsplit('_', 1)[0]
                dicom_filepath = find(dicom_filename, path_to_dicom_folders)
                if dicom_filepath is not None:
                    print(dicom_filepath)
                # print(os.path.join(os.path.split(path_to_dicom_folders)[0], '2DStrain_atrium'))
                move(dicom_filepath,
                     os.path.join(os.path.split(path_to_dicom_folders)[0], '2DStrain_atrium', dicom_filename))
        except IndexError:
            exit('Moving strain completed')


if __name__ == '__main__':
    path_to_dicoms  = r'D:\PredictAF_nonanon\HTcopy\HT_done'
    path_to_dicoms2 = r'D:\PredictAF_nonanon\HTcopy\2DStrain'
    path_to_dicoms_test = r'F:\ABC0455'
    path_to_dicoms3 = r'C:\Users\mm18\Downloads\PV002 Echo 1 - 3D volume exports'
    path_to_dicoms4 = r'C:\EchoPAC_PC\ARCHIVE\Export\GEMS_IMG\2019_SEP\16\_P131238'

    move_atrial_strain_files(path_to_dicoms2, r'D:\2DS_ROI_ES_HT')

    # compute_gradient(img)
    # read_dicom(path_to_dicoms4)
    # copy_2ds_sequences(path_to_dicoms)
    # remove_modified_dicoms(path_to_dicoms2)
