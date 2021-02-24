import os
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from src.models.fetch_data_2d import get_tf_dataset


project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
print(f"c'est le dotenv {dotenv_path}")
dotenv.load_dotenv(str(dotenv_path))

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

path_output = project_dir / "data/processed/2d"

path_output.mkdir(parents=True, exist_ok=True)

image_size = (224, 224)
n_epochs = 1000


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

    data_test = get_tf_dataset(
        patient_test,
        path_data_nii,
        path_mask_lung_nii,
        output_shape=image_size,
        return_patient_name=True,
    ).as_numpy_iterator()
    n_test = len(patient_test)
    n_train = len(patient_train)

    path_file_test = ((path_output / 'test.hdf5').resolve())
    path_file_test.unlink(missing_ok=True)  # delete file if exists
    f_test = h5py.File(path_file_test, 'a')
    epoch = 0
    for i, (image, mask, patient) in tqdm(enumerate(data_test)):
        if i % n_test == 0:
            epoch += 1
            f_test.create_group(f"epoch_{epoch}")
        patient_name = patient.decode("utf-8")
        h, w, c = image.shape
        f_test.create_group(f"epoch_{epoch}/{patient_name}")
        f_test.create_dataset(f"epoch_{epoch}/{patient_name}/image",
                              data=image,
                              dtype="float32")
        f_test.create_dataset(f"epoch_{epoch}/{patient_name}/mask",
                              data=mask,
                              dtype="uint16")

    f_test.close()

    data_train = get_tf_dataset(
        patient_train,
        path_data_nii,
        path_mask_lung_nii,
        output_shape=image_size,
        random_center=True,
        augment_angles=(45, 45, 90),
        return_patient_name=True,
    ).repeat(n_epochs).as_numpy_iterator()

    path_file_train = ((path_output / 'train.hdf5').resolve())
    path_file_train.unlink(missing_ok=True)  # delete file if exists
    f_train = h5py.File(path_file_train, 'a')
    epoch = 0
    for i, (image, mask, patient) in tqdm(enumerate(data_train)):
        if i % n_train == 0:
            epoch += 1
            f_train.create_group(f"epoch_{epoch}")
        patient_name = patient.decode("utf-8")
        h, w, c = image.shape
        f_train.create_group(f"epoch_{epoch}/{patient_name}")
        f_train.create_dataset(f"epoch_{epoch}/{patient_name}/image",
                               data=image,
                               dtype="float32")
        f_train.create_dataset(f"epoch_{epoch}/{patient_name}/mask",
                               data=mask,
                               dtype="uint16")

    f_train.close()


if __name__ == '__main__':
    # project_dir = Path(__file__).resolve().parents[2]
    main()
