import os
from pathlib import Path
from random import shuffle
from itertools import product

import dotenv
import tensorflow as tf
import h5py
import pandas as pd

from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import unet_model, CustomModel
from src.models.losses_2d import dice_coe_1_hard

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

bs = 4
n_epochs = 1
n_prefetch = 20
image_size = (256, 256)


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


alphas = [0.25, 0.5, 0.75, 1.0]
ws_gtvl = [2, 4]
ws_gtvt = [1, 2]
ws_lung = [1]


def main():
    file_train = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d_pet_normalized/train.hdf5",
        "r")
    # file_test = h5py.File(
    #     "/home/val/python_wkspce/plc_seg/data/processed/2d_pet_normalized/test.hdf5",
    #     "r")
    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")
    patient_list = list(file_train.keys())
    patient_list = [p for p in patient_list if p not in ["PatientLC_63"]]
    patient_list_train, patient_list_val = get_trainval_patient_list(
        clinical_df, patient_list)

    data_val = get_tf_data(
        file_train,
        clinical_df,
        output_shape=(256, 256),
        random_slice=False,
        centered_on_gtvt=True,
        patient_list=patient_list_val,
    ).cache().batch(bs)
    data_train = get_tf_data(file_train,
                             clinical_df,
                             output_shape=(256, 256),
                             random_slice=True,
                             random_shift=20,
                             n_repeat=10,
                             num_parallel_calls='auto',
                             oversample_plc_neg=True,
                             patient_list=patient_list_train).batch(bs)
    for alpha, w_gtvl, w_gtvt, w_lung in product(alphas, ws_gtvl, ws_gtvt,
                                                 ws_lung):
        model_name = (f"model_{alpha:0.2f}_alpha_{w_gtvl:0.2f}_wplc_"
                      f"{w_gtvt:0.2f}_wt_{w_lung:0.2f}_wt".replace(".", "-"))
        model_path = project_dir / f"models/{model_name}/epochs_{n_epochs}/"
        checkpoint_filepath = project_dir / (f"models/{model_name}"
                                             f"/checkpoint/checkpoint")
        results_path = project_dir / f"data/results/{model_name}.csv"
        results_df = pd.DataFrame()
        model_ = unet_model(3, input_shape=image_size + (3, ))
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        model = CustomModel(
            model_.inputs,
            model_.outputs,
            alpha=alpha,
            w_lung=w_lung,
            w_gtvt=w_gtvt,
            w_gtvl=w_lung,
        )
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            # save_freq=1,
        )

        model.compile(
            optimizer=optimizer,
            run_eagerly=True,
        )

        model.fit(
            x=data_train,
            epochs=1,  # n_epochs,
            validation_data=data_val,
            callbacks=[model_checkpoint_callback],
        )
        res_train = model.evaluate(x=data_train)
        res_val = model.evaluate(x=data_val)
        results_df.append(res_train, ignore_index=True)
        results_df.append(res_val, ignore_index=True)
        results_df.to_csv(results_path)

        model.save(model_path)
    file_train.close()
    # file_test.close()


if __name__ == '__main__':
    main()