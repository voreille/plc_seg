import os
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.fetch_data import get_tf_dataset, tf_parse_image, parse_image
from src.models.models import Unet, UnetLight

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])


def main():
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

    data_test = get_tf_dataset(patient_test, path_data_nii, path_mask_lung_nii)
    data_train = get_tf_dataset(patient_train,
                                path_data_nii,
                                path_mask_lung_nii,
                                random_center=True,
                                augment_angles=(45, 45, 90))
    model = UnetLight()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer,
                  loss=None,
                  metrics=None,
                  loss_weights=None,
                  weighted_metrics=None,
                  run_eagerly=None,
                  steps_per_execution=None,
                  **kwargs)

    model.fit(x=data_train, batch_size=1, epochs=1, validation_split=0.1)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]

    dotenv_path = project_dir / ".env"
    dotenv.load_dotenv(str(dotenv_path))
    main()