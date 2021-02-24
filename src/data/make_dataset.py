import os
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import SimpleITK as sitk

from src.models.fetch_data import (get_ct_range, normalize_image,
                                   split_lung_mask)

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"

dotenv.load_dotenv(str(dotenv_path))

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

path_output = project_dir / "data/processed/2d"

path_output.mkdir(parents=True, exist_ok=True)

image_size = (224, 224)
n_epochs = 1000


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def get_bb_mask_voxel(mask_sitk):
    mask = sitk.GetArrayFromImage(mask_sitk)
    positions = np.where(mask != 0)
    z_min = np.min(positions[0])
    y_min = np.min(positions[1])
    x_min = np.min(positions[2])
    z_max = np.max(positions[0])
    y_max = np.max(positions[1])
    x_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, z_max, y_max


def get_bb_mask_mm(mask_sitk):
    x_min, y_min, z_min, x_max, z_max, y_max = get_bb_mask_voxel(mask_sitk)
    return (*mask_sitk.TransformIndexToPhysicalPoint(
        [int(x_min), int(y_min), int(z_min)]),
            *mask_sitk.TransformIndexToPhysicalPoint(
                [int(x_max), int(y_max), int(z_max)]))


def parse_image(
    patient,
    path_nii,
    path_lung_mask_nii,
    output_shape=(512, 512),
    spacing=(1, 1, 1),
    ct_window_str="lung",
    augment_mirror=False,
    interp_order=3,
    mask_smoothing=False,
    smoothing_radius=3,
):
    """Parse the raw data of HECKTOR 2020

    Args:
        folder_name ([Path]): the path of the folder containing 
        the 3 sitk images (ct, pt and mask)
    """
    output_shape = np.array(output_shape)
    patient_name = patient
    # t1 = time.time()
    ct_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__CT.nii.gz")).resolve()))
    pt_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__PT.nii.gz")).resolve()))
    mask_gtvt_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_T__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_gtvl_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_L__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_lung_sitk = sitk.ReadImage(
        str((path_lung_mask_nii /
             (patient_name + "__LUNG__SEG__CT.nii.gz")).resolve()))
    # print(f"Time reading the files for patient {patient} : {time.time()-t1}")
    # t1 = time.time()
    resampler = sitk.ResampleImageFilter()
    if interp_order == 3:
        resampler.SetInterpolator(sitk.sitkBSpline)
    # compute center
    bb_gtvt = get_bb_mask_mm(mask_gtvt_sitk)
    bb_gtvl = get_bb_mask_mm(mask_gtvl_sitk)
    z_max = np.max([bb_gtvt[-1], bb_gtvl[-1]])
    z_min = np.min([bb_gtvt[2], bb_gtvl[2]])
    origin = np.array(ct_sitk.GetOrigin())
    origin[2] = z_min
    z_shape = int((z_max - z_min) / spacing[2])

    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize((int(output_shape[0]), int(output_shape[1]), z_shape))

    ct_sitk = resampler.Execute(ct_sitk)
    pt_sitk = resampler.Execute(pt_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_gtvt_sitk = resampler.Execute(mask_gtvt_sitk)
    mask_gtvl_sitk = resampler.Execute(mask_gtvl_sitk)
    mask_lung_sitk = resampler.Execute(mask_lung_sitk)
    mask_lung1_sitk, mask_lung2_sitk = split_lung_mask(mask_lung_sitk)
    if mask_smoothing:
        smoother = sitk.BinaryMedianImageFilter()
        smoother.SetRadius(int(smoothing_radius))
        mask_gtvt_sitk = smoother.Execute(mask_gtvt_sitk)
        mask_gtvl_sitk = smoother.Execute(mask_gtvl_sitk)
        mask_lung1_sitk = smoother.Execute(mask_lung1_sitk)
        mask_lung2_sitk = smoother.Execute(mask_lung2_sitk)
    mask_gtvt = to_np(mask_gtvt_sitk)
    mask_gtvl = to_np(mask_gtvl_sitk)
    mask_lung1 = to_np(mask_lung1_sitk)
    mask_lung2 = to_np(mask_lung2_sitk)

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)
    hu_low, hu_high = get_ct_range(ct_window_str)
    ct[ct > hu_high] = hu_high
    ct[ct < hu_low] = hu_low
    ct = (2 * ct - hu_high - hu_low) / (hu_high - hu_low)

    # pt = normalize_image(pt)

    image = np.stack([ct, pt, np.zeros_like(ct)], axis=-1)
    mask = np.stack([mask_gtvt, mask_gtvl, mask_lung1, mask_lung2], axis=-1)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0

    # print(f"Time preprocessing for patient {patient} : {time.time()-t1}")
    return image, mask


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    patient_list = [
        f.name.split("__")[0] for f in path_mask_lung_nii.rglob("*LUNG*")
    ]
    clinical_df = pd.read_excel(path_clinical_info)
    clinical_df["PatientID"] = clinical_df["patient_id"].map(
        lambda x: "PatientLC_" + str(x))
    patients_test = clinical_df[clinical_df["is_chuv"] == 0]["PatientID"]
    patient_test = [p for p in patients_test if p in patient_list]

    patients_train = clinical_df[clinical_df["is_chuv"] == 1]["PatientID"]
    patient_train = [p for p in patients_train if p in patient_list]

    path_file_test = ((path_output / 'test.hdf5').resolve())
    path_file_test.unlink(missing_ok=True)  # delete file if exists
    f_test = h5py.File(path_file_test, 'a')
    for patient in tqdm(patient_test):

        image, mask = parse_image(patient, path_data_nii, path_mask_lung_nii)

        f_test.create_group(f"{patient}")
        f_test.create_dataset(f"{patient}/image", data=image, dtype="float32")
        f_test.create_dataset(f"{patient}/mask", data=mask, dtype="uint16")

    f_test.close()

    path_file_train = ((path_output / 'train.hdf5').resolve())
    path_file_train.unlink(missing_ok=True)  # delete file if exists
    f_test = h5py.File(path_file_train, 'a')
    for patient in tqdm(patient_train):

        image, mask = parse_image(patient, path_data_nii, path_mask_lung_nii)

        f_test.create_group(f"{patient}")
        f_test.create_dataset(f"{patient}/image", data=image, dtype="float32")
        f_test.create_dataset(f"{patient}/mask", data=mask, dtype="uint16")

    f_test.close()


if __name__ == '__main__':
    # project_dir = Path(__file__).resolve().parents[2]
    main()
