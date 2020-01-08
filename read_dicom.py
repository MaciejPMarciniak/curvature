import pydicom
from pydicom.tag import Tag
from dicom_contour.contour import create_image_mask_files, get_contour_file
import os
import struct
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

    skipped = 0
    for dirpath, dirnames, filenames in os.walk(path_to_dicom_folders):
        pid = True

        for dicom_filename in filenames:
            ds = pydicom.read_file(os.path.join(dirpath, dicom_filename))
            if pid:
                i += 1
                print(i)
                print('Case: {}'.format(ds.PatientID))
                print('Filename: {}'.format(os.path.split(dirpath)[-1]))
                print(ds.ImageType)
                patient_ids.append(ds.PatientID)
                patient_files.append(dirpath)
                pid = False

            try:
                if ds.ImageType[6] == 'GEMS2DSTRAIN':
                    print(dicom_filename)
                    print(ds.ImageType)
                    two_ds_fol = _check_directory(os.path.join(two_ds_dir, ds.PatientID))
                    shutil.copy(os.path.join(dirpath, dicom_filename),
                                os.path.join(two_ds_fol, dicom_filename+'_2DS'))
            except IndexError:
                skipped += 1
                print(ds.ImageType)
                print('missing ImageType index, skipping')
                continue
    print('skipped: {}'.format(skipped))
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
            # if roi_data[int(roi_data.shape[0]/2), 0] > np.mean((roi_data[0, 0], roi_data[-1, 0])):
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


def read_dicom(path_to_dicom_folders):
    for (dirpath, dirnames, filenames) in os.walk(path_to_dicom_folders):
        print(filenames)
        for dicom_filename in filenames:
            if dicom_filename == '1.2.528.1.1003.1.11526562807450000000546900218626.1.1.dcm':

                ds = pydicom.read_file(os.path.join(dirpath, dicom_filename))
                print(ds)
                tag = Tag(0x79051011)
                tag2 = Tag(0x79051013)
                tag3 = Tag(0x79051017)
                tag4 = Tag(0x80051021)
                pdb = ds[tag].value[0]
                pdb2 = pdb[tag2].value[0]
                pdb3 = pdb2[tag3].value[0]
                pdb4 = pdb3[tag4].value
                print('$$$$$$$$$$$$')
                pdb = np.fromstring(pdb4, dtype='int8')
                print(pdb)
                print(len(pdb))
                # print(pdb4)
                #
                # print(ds)
                # bits_allocated = str(ds.BitsAllocated)
                # pixel_spacing = ds.PixelSpacing
                # print(pdb)
                # pdb = np.fromstring(pdb, dtype='int8')
                # print(pdb)
                # print(len(pdb))
                pa = np.fromstring(pdb, dtype='int8').reshape((ds.Rows, ds.Columns))
                plt.imshow(pa)
                plt.show()
                # # plt.savefig('pixel_array.png')

                # print(ds)


def print_dicom_folders(path_to_dicoms):
    relevant_ids = ['aduheart013', 'aduheart018', 'aduheart021', 'aduheart033', 'aduheart185', 'aduheart452',
                    'aduheart196', 'aduheart198', 'aduheart327', 'aduheart334']
    relevant_ids = [relevant_id.upper() for relevant_id in relevant_ids]
    print(relevant_ids)

    for (dirpath, dirfolder, files) in os.walk(path_to_dicoms):
        for dicom_file in files:

            dfil = pydicom.read_file(os.path.join(dirpath, dicom_file))
            print(dfil.PatientID)
            if dfil.PatientID in relevant_ids:
                print('Found!!!',  dfil.PatientID)
                print(dirpath)
            break

if __name__ == '__main__':
    path_to_dicoms  = r'G:\HospitalClinic\CurvatureStudy\AduHeart-RAW-NO-STRAIN'
    path_to_dicoms2 = r'C:\Data\ProjectDevelopmentalAtlas\GenerationRTest\R113735'
    path_to_dicoms_test = r'F:\ABC0455'
    path_to_dicoms3 = r'C:\Users\mm18\Downloads\PV002 Echo 1 - 3D volume exports'
    path_to_dicoms4 = r'C:\EchoPAC_PC\ARCHIVE\Export\GEMS_IMG\2019_SEP\16\_P131238'

    # move_atrial_strain_files(path_to_dicoms2, r'D:\2DS_ROI_ES_HT')

    print_dicom_folders(r'G:\HospitalClinic\CurvatureStudy\AduHeart-DICOM')
    # read_dicom(path_to_dicoms2)
    # copy_2ds_sequences(path_to_dicoms2)
    # remove_modified_dicoms(path_to_dicoms2)
