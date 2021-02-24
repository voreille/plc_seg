import os
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.fetch_data_2d_fixed import get_tf_dataset
from src.models.models_2d import unet_model
from src.models.losses_2d import dice_coe_loss, dice_coe_hard

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

bs = 1
n_epochs = 10
n_prefetch = 20
image_size = (256, 256)


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

    data_test = get_tf_dataset(
        patient_test,
        path_data_nii,
        path_mask_lung_nii,
        num_parallel_calls=15,
        output_shape=image_size,
    ).batch(bs)

    data_train = get_tf_dataset(
        patient_train,
        path_data_nii,
        path_mask_lung_nii,
        output_shape=image_size,
        num_parallel_calls=15,
        # random_center=True,
        # augment_angles=(45, 45, 90)
    ).batch(bs)
    model = unet_model(3, input_shape=image_size + (3, ))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=dice_coe_loss,
        metrics=dice_coe_hard,
    )

    model.fit(x=data_train, epochs=1000, validation_data=data_test)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]

    dotenv_path = project_dir / ".env"
    dotenv.load_dotenv(str(dotenv_path))
    main()