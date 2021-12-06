import os
from pathlib import Path
from random import shuffle
import datetime

import dotenv
import tensorflow as tf
import kerastuner as kt
import h5py
import pandas as pd

from src.models.losses_2d import CustomLoss, gtvl_loss
from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import unet_model

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))
log_dir = project_dir / ("logs/fit/" +
                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
path_model = project_dir / "models/clean_model/model__a_0-75__splc_40__wplc_1__wt_0__wl_0"

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

bs = 4
n_epochs = 200
n_prefetch = 20
image_size = (256, 256)


def model_builder(hp):
    model = unet_model(3, input_shape=image_size + (3, ))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3])
    hp_alpha = hp.Choice('alpha', values=[0.75])
    hp_sgtvl = hp.Choice('s_gtvl', values=[40])
    hp_wgtvl = hp.Choice('w_gtvl', values=[1])
    hp_wgtvt = hp.Choice('w_gtvt', values=[0])
    hp_wlung = hp.Choice('w_lung', values=[0])

    def gtvl_metric(y_true, y_pred):
        return gtvl_loss(y_true, y_pred, scaling=1.0, pos_weight=0.75)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=CustomLoss(pos_weight=hp_alpha,
                        w_lung=hp_wlung,
                        w_gtvt=hp_wgtvt,
                        w_gtvl=hp_wgtvl,
                        s_gtvl=hp_sgtvl),
        metrics=[gtvl_metric])

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
        output_shape_image=(256, 256),
        random_slice=False,
        label_to_contain="GTV T",
        patient_list=patient_list_val,
    ).cache().batch(2)
    data_train = get_tf_data(file_train,
                             clinical_df,
                             output_shape_image=(256, 256),
                             random_slice=True,
                             random_shift=20,
                             n_repeat=10,
                             num_parallel_calls='auto',
                             oversample_plc_neg=True,
                             patient_list=patient_list_train).batch(bs)

    tuner = kt.Hyperband(model_builder,
                         objective=kt.Objective("val_gtvl_metric",
                                                direction="min"),
                         max_epochs=50,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=0,
        mode='auto',
        restore_best_weights=True)
    tuner.search(x=data_train,
                 epochs=50,
                 validation_data=data_val,
                 callbacks=[early_stop_callback])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal
     hyperparameters are alpha: {best_hps.get('alpha')},
     s_gtvl: {best_hps.get('s_gtvl')},
     w_gtvl: {best_hps.get('w_gtvl')},
     w_gtvt: {best_hps.get('w_gtvt')},
     w_lung: {best_hps.get('w_lung')},
     and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    model = tuner.hypermodel.build(best_hps)

    model.fit(
        x=data_train,
        epochs=n_epochs,
        validation_data=data_val,
        callbacks=[early_stop_callback, tensorboard_callback],
    )

    model.save(path_model)
    file_train.close()


if __name__ == '__main__':
    main()
