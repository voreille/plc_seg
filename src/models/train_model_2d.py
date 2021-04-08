import os
from pathlib import Path
from random import shuffle
import random
import datetime

import dotenv
import tensorflow as tf
import h5py
import pandas as pd
from numpy.random import seed

from src.models.losses_2d import CustomLoss, gtvl_loss
from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import unet_model

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))
log_dir = project_dir / ("logs/fit/" +
                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
path_model = project_dir / "models/clean_model/model_gtvl_gtvt"

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

bs = 4
n_epochs = 10
n_prefetch = 20
image_size = (256, 256)

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)
tf.random.set_seed(2)


def model_builder(
    lr=1e-3,
    alpha=0.5,
    s_gtvl=40.0,
    w_gtvl=1.0,
    w_gtvt=0.0,
    w_lung=0.0,
):
    model = unet_model(3, input_shape=image_size + (3, ))

    def gtvl_metric(y_true, y_pred):
        return gtvl_loss(y_true, y_pred, scaling=s_gtvl, pos_weight=alpha)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=CustomLoss(pos_weight=alpha,
                        w_lung=w_lung,
                        w_gtvt=w_gtvt,
                        w_gtvl=w_gtvl,
                        s_gtvl=s_gtvl),
        metrics=[gtvl_metric],
        run_eagerly=False,
    )

    return model


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
        output_shape=(256, 256),
        random_slice=False,
        label_to_center="GTV T",
        patient_list=patient_list_val,
    ).cache().batch(2)
    data_train = get_tf_data(file_train,
                             clinical_df,
                             output_shape=(256, 256),
                             random_slice=True,
                             random_shift=20,
                             n_repeat=10,
                             num_parallel_calls='auto',
                             oversample_plc_neg=True,
                             patient_list=patient_list_train).batch(bs)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=0,
        mode='min',
        restore_best_weights=True)
    model = model_builder(alpha=0.5,
                          s_gtvl=40.0,
                          w_gtvl=1.0,
                          w_gtvt=1.0,
                          w_lung=0.0)
    model.fit(
        x=data_train,
        epochs=n_epochs,
        validation_data=data_val,
        # callbacks=[early_stop_callback, tensorboard_callback],
    )

    model.save(path_model)
    file_train.close()


if __name__ == '__main__':
    main()
