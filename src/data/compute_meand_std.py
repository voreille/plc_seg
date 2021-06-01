from pathlib import Path

import dotenv
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"

dotenv.load_dotenv(str(dotenv_path))

path_data_nii = Path(
    "/home/val/python_wkspce/plc_seg/data/interim/nii_resampled")
path_mask_lung_nii = Path(
    "/home/val/python_wkspce/plc_seg/data/interim/nii_resampled")

path_output = project_dir / "data"

path_output.mkdir(parents=True, exist_ok=True)


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def get_bb_mask_voxel(mask):
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, y_max, z_max


def get_bb_mask_sitk_voxel(mask_sitk):
    mask = to_np(mask_sitk)
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, y_max, z_max


def get_bb_mask_mm(mask_sitk):
    x_min, y_min, z_min, x_max, y_max, z_max = get_bb_mask_sitk_voxel(
        mask_sitk)
    return (*mask_sitk.TransformIndexToPhysicalPoint(
        [int(x_min), int(y_min), int(z_min)]),
            *mask_sitk.TransformIndexToPhysicalPoint(
                [int(x_max), int(y_max), int(z_max)]))


def slice_volumes(*args, s1=0, s2=-1):
    output = []
    for im in args:
        output.append(im[:, :, s1:s2 + 1])

    return output


def parse_image(
    patient_name,
    path_nii,
    path_lung_mask_nii,
):
    # t1 = time.time()
    ct_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__CT.nii.gz")).resolve()))
    pt_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__PT.nii.gz")).resolve()))
    mask_lung_sitk = sitk.ReadImage(
        str((path_lung_mask_nii /
             (patient_name + "__LUNG__SEG__CT.nii.gz")).resolve()))

    mask_lung = to_np(mask_lung_sitk)

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)

    bb = get_bb_mask_voxel(mask_lung)
    z_max = bb[-1]
    z_min = bb[2]
    ct, pt = slice_volumes(ct, pt, s1=z_min, s2=z_max)

    patient_id = int(patient_name.split("_")[-1])
    df = pd.DataFrame({
        "patient_id": [patient_id],
        "mean_ct": [np.mean(ct)],
        "std_ct": [np.std(ct)],
        "mean_pt": [np.mean(pt)],
        "std_pt": [np.std(pt)],
    })

    return df


def main():
    patient_list = [
        f.name.split("__")[0] for f in path_mask_lung_nii.rglob("*LUNG*")
    ]

    path_file = ((path_output / 'mean_std.csv').resolve())
    path_file.unlink(missing_ok=True)  # delete file if exists
    df = pd.DataFrame()
    for patient in tqdm(patient_list):

        tmp_df = parse_image(patient, path_data_nii, path_mask_lung_nii)
        df = df.append(tmp_df, ignore_index=True)

    df.set_index("patient_id").to_csv(path_file)


if __name__ == '__main__':
    # project_dir = Path(__file__).resolve().parents[2]
    main()
