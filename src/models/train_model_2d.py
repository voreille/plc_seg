import os
from pathlib import Path
from random import shuffle
import datetime

import dotenv
import tensorflow as tf
import h5py
import pandas as pd
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from src.models.losses_2d import CustomLoss, gtvl_loss
from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import (OUTPUT_CHANNELS, UnetClassif, unet_model,
                                  unetclassif_model, classif_model, Unet,
                                  UnetClassif)
from src.models.evaluation import PLCSegEvaluator, MultiTaskEvaluator, ClassificationEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))
log_dir = project_dir / ("logs/fit/" +
                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
path_model = project_dir / "models/clean_model/model1"

path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

bs = 32
n_epochs = 200
n_prefetch = 20
image_size = (256, 256)

task_type = "segmentation"  # segmentation+plc_status", "plc_status"
alpha = 0.75
w_gtvl = 1
s_gtvl = 1
w_gtvt = 0
w_lung = 0

model_dict = {
    "segmentation": unet_model,
    "plc_status": classif_model,
    "segmentation+plc_status": unetclassif_model
}
losses_dict = {
    "segmentation": [
        CustomLoss(alpha=alpha,
                   w_lung=w_lung,
                   w_gtvt=w_gtvt,
                   w_gtvl=w_gtvl,
                   s_gtvl=s_gtvl,
                   loss_type="masked"),
    ],
    "segmentation+plc_status": [
        CustomLoss(alpha=alpha,
                   w_lung=w_lung,
                   w_gtvt=w_gtvt,
                   w_gtvl=w_gtvl,
                   s_gtvl=s_gtvl,
                   loss_type="masked"),
        tf.keras.losses.SparseCategoricalCrossentropy(),
    ],
    "plc_status": [
        tf.keras.losses.SparseCategoricalCrossentropy(),
    ],
}

evaluator_dict = {
    "segmentation": PLCSegEvaluator,
    "plc_status": ClassificationEvaluator,
    "segmentation+plc_status": MultiTaskEvaluator,
}


def get_split_patient_lists(df, patient_list):
    id_list = [int(p.split('_')[1]) for p in patient_list]
    df = df.loc[id_list, :]
    id_patient_plc_neg_training = list(df[(df["is_chuv"] == 1)
                                          & (df["plc_status"] == 0)].index)
    id_patient_plc_pos_training = list(df[(df["is_chuv"] == 1)
                                          & (df["plc_status"] == 1)].index)
    shuffle(id_patient_plc_neg_training)
    shuffle(id_patient_plc_pos_training)
    id_patient_plc_neg_val = id_patient_plc_neg_training[:5]
    id_patient_plc_pos_val = id_patient_plc_pos_training[:13]
    id_val = id_patient_plc_neg_val + id_patient_plc_pos_val
    id_patient_plc_neg_train = id_patient_plc_neg_training[5:]
    id_patient_plc_pos_train = id_patient_plc_pos_training[13:]
    id_train = id_patient_plc_neg_train + id_patient_plc_pos_train

    patient_list_val = [f"PatientLC_{i}" for i in id_val]
    patient_list_train = [f"PatientLC_{i}" for i in id_train]
    patient_list_test = [
        p for p in patient_list
        if p not in patient_list_val and p not in patient_list_train
    ]
    return patient_list_train, patient_list_val, patient_list_test


def main():
    file = h5py.File(
        "/home/valentin/python_wkspce/plc_seg/data/processed/2d_pet_normalized/data.hdf5",
        "r")
    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")
    patient_list = list(file.keys())
    patient_list = [p for p in patient_list if p not in ["PatientLC_63"]]
    patient_list_train, patient_list_val, patient_list_test = get_split_patient_lists(
        clinical_df, patient_list)

    data_val = get_tf_data(
        file,
        clinical_df,
        output_shape=(256, 256),
        random_slice=False,
        centered_on_gtvt=False,
        patient_list=patient_list_val,
        output_type=task_type,  # "segmentation" "plc_status"
    ).cache().batch(2)
    data_train = get_tf_data(
        file,
        clinical_df,
        output_shape=(256, 256),
        random_slice=True,
        random_shift=20,
        n_repeat=10,
        num_parallel_calls='auto',
        oversample_plc_neg=True,
        output_type=task_type,  # "segmentation" "plc_status"
        patient_list=patient_list_train,
        random_angle=15,
    ).batch(bs)

    model = model_dict[task_type](3, input_shape=image_size + (3, ))
    # model = UnetClassif()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    def gtvl_metrics(y_true, y_pred):
        return gtvl_loss(y_true, y_pred, scaling=s_gtvl, alpha=alpha)

    model.compile(
        loss=losses_dict[task_type],
        # metrics=[gtvl_metrics],
        optimizer=optimizer,
        run_eagerly=False,
    )

    evaluator = evaluator_dict[task_type](
        model,
        file,
        clinical_df,
        output_shape=(256, 256),
    )
    auc_train = evaluator(patient_list_train)
    auc_val = evaluator(patient_list_val)
    auc_test = evaluator(patient_list_test)
    print(f"======================== "
          f"That's the AUC before fitting"
          f" of the train, val and test resp.: "
          f"{auc_train}, {auc_val} and {auc_test}")

    model.fit(
        x=data_train,
        epochs=n_epochs,
        validation_data=data_val,
        callbacks=[early_stop_callback, tensorboard_callback],
    )
    auc_train = evaluator(
        patient_list_train,
        path_to_save_nii=
        "/home/valentin/python_wkspce/plc_seg/data/reports/train")
    auc_val = evaluator(
        patient_list_val,
        path_to_save_nii="/home/valentin/python_wkspce/plc_seg/data/reports/val"
    )
    auc_test = evaluator(
        patient_list_test,
        path_to_save_nii=
        "/home/valentin/python_wkspce/plc_seg/data/reports/test")
    print(f"That's the AUC after fitting"
          f" of the train, val and test resp.: "
          f"{auc_train}, {auc_val} and {auc_test}")

    # model.save(path_model)
    file.close()


if __name__ == '__main__':
    main()
