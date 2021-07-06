import os
from pathlib import Path
from random import shuffle
import datetime

import dotenv
import h5py
import pandas as pd

from src.models.fetch_data_from_hdf5 import get_tf_data

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))
log_dir = project_dir / ("logs/fit/" +
                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

image_size = (256, 256)
bs = 2


def get_trainval_patient_list(df, patient_list):
    id_list = [int(p.split('_')[1]) for p in patient_list]
    df = df.loc[id_list, :]
    id_patient_plc_neg_training = list(df[(df["is_chuv"] == 1)
                                          & (df["plc_status"] == 0)].index)
    id_patient_plc_pos_training = list(df[(df["is_chuv"] == 1)
                                          & (df["plc_status"] == 1)].index)
    shuffle(id_patient_plc_neg_training)
    shuffle(id_patient_plc_pos_training)
    id_patient_plc_neg_val = id_patient_plc_neg_training[:2]
    id_patient_plc_pos_val = id_patient_plc_pos_training[:4]
    id_val = id_patient_plc_neg_val + id_patient_plc_pos_val
    id_patient_plc_neg_train = id_patient_plc_neg_training[2:]
    id_patient_plc_pos_train = id_patient_plc_pos_training[4:]
    id_train = id_patient_plc_neg_train + id_patient_plc_pos_train

    patient_list_val = [f"PatientLC_{i}" for i in id_val]
    patient_list_train = [f"PatientLC_{i}" for i in id_train]
    return patient_list_train, patient_list_val


def main():
    file_train = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d_pet_normalized/train.hdf5",
        "r")
    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")
    patient_list = list(file_train.keys())
    patient_list = [p for p in patient_list if p not in ["PatientLC_63"]]
    patient_list_train, patient_list_val = get_trainval_patient_list(
        clinical_df, patient_list)

    data_val = get_tf_data(
        file_train,
        clinical_df,
        output_shape_image=(256, 256),
        random_slice=False,
        centered_on_gtvt=True,
        patient_list_copy=patient_list_val,
    ).cache().batch(2)
    data_train = get_tf_data(file_train,
                             clinical_df,
                             output_shape_image=(256, 256),
                             random_slice=True,
                             random_shift=20,
                             n_repeat=10,
                             num_parallel_calls='auto',
                             oversample_plc_neg=True,
                             patient_list_copy=patient_list_train).batch(bs)

    for x, y, plc_status in data_val.as_numpy_iterator():
        print(
            f"voici, voilé le x {x.shape}, le y {y.shape} et le plc_status {plc_status}"
        )

    file_train.close()


if __name__ == '__main__':
    main()
