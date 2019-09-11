import pydicom
import os
import shutil
import csv


def _check_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


def copy_2ds_sequences(path_to_dicom_folders):

    two_ds_dir = _check_directory(os.path.join(os.path.split(path_to_dicom_folders)[0], '2DStrain'))

    # folders = ['EG162528', 'EG183845', 'EM172205', 'EP173952', 'JP184628',
    # 'LS195554', 'MB194510', 'MM164245', 'MV162659']
    i = 0
    patient_ids = []
    patient_files = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_dicom_folders):
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
                two_ds_fol = _check_directory(os.path.join(two_ds_dir, os.path.split(dirpath)[-1]+'_2DS'))
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


if __name__ == '__main__':
    path_to_dicoms = 'D:\PREDICT-AF'
    path_to_dicoms2 = 'F:\PredictAF_nonanon\HTcopy\HT_done'
    ptd3 = r'D:\PredictAF_nonanon\ath_2_done\martinez_nieto\GEMS_IMG\2017_MAY\08\CM142217'
    copy_2ds_sequences(path_to_dicoms2)
    # remove_modified_dicoms(path_to_dicoms2)
