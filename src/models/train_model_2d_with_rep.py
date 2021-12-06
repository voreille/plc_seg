import os
from pathlib import Path
from random import shuffle
import datetime

import click
import dotenv
import tensorflow as tf
import h5py
import pandas as pd
# from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from src.models.losses_2d import CustomLoss, gtvl_loss
from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import (OUTPUT_CHANNELS, UnetClassif, unet_model,
                                  unetclassif_model, classif_model, Unet,
                                  UnetClassif)
from src.models.evaluation import PLCSegEvaluator, MultiTaskEvaluator, ClassificationEvaluator

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


def train_one_rep(model, losses, file, clinical_df, task_type):
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
        # random_angle=0,
    ).batch(bs)

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

    model.compile(
        loss=losses,
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
    model.fit(
        x=data_train,
        epochs=n_epochs,
        validation_data=data_val,
        callbacks=[early_stop_callback, tensorboard_callback],
    )
    auc_train = evaluator(patient_list_train)
    auc_val = evaluator(patient_list_val)
    auc_test = evaluator(patient_list_test)
    print(f"That's the AUC after fitting"
          f" of the train, val and test resp.: "
          f"{auc_train}, {auc_val} and {auc_test}")
    return auc_train, auc_val, auc_test

    # model.save(path_model)


@click.command()
@click.option('--n_rep', type=click.INT, default=3)
@click.option('--alpha', type=click.FLOAT, default=0.75)
@click.option('--w_gtvl', type=click.FLOAT, default=1.0)
@click.option('--s_gtvl', type=click.FLOAT, default=1.0)
@click.option('--w_gtvt', type=click.FLOAT, default=0.0)
@click.option('--w_lung', type=click.FLOAT, default=0.0)
@click.option('--gpu_id', type=click.STRING, default="0")
@click.option('--task_type', type=click.STRING, default="segmentation")
def main(alpha, w_gtvl, s_gtvl, w_gtvt, w_lung, n_rep, gpu_id, task_type):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # task_type = "segmentation"  # segmentation+plc_status", "plc_status"
    params = {
        "alpha": alpha,
        "w_gtvl": w_gtvl,
        "s_gtvl": s_gtvl,
        "w_gtvt": w_gtvt,
        "w_lung": w_lung,
    }
    model_dict = {
        "segmentation": unet_model,
        "plc_status": classif_model,
        "segmentation+plc_status": unetclassif_model
    }
    losses_dict = {
        "segmentation": [
            CustomLoss(loss_type="masked", **params),
        ],
        "segmentation+plc_status": [
            CustomLoss(loss_type="masked", **params),
            tf.keras.losses.SparseCategoricalCrossentropy(),
        ],
        "plc_status": [
            tf.keras.losses.SparseCategoricalCrossentropy(),
        ],
    }

    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")
    file = h5py.File(
        "/home/valentin/python_wkspce/plc_seg/data/processed/2d_pet_normalized/data.hdf5",
        "r")
    results_df = pd.DataFrame(columns=pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["split", "type"]))
    print(f"OUAI MEC JUST TO CHECK ========= les params c'est {params}")
    for k in range(n_rep):
        model = model_dict[task_type](3, input_shape=image_size + (3, ))
        auc_train, auc_val, auc_test = train_one_rep(model,
                                                     losses_dict[task_type],
                                                     file, clinical_df,
                                                     task_type)
        for key, val in auc_train.items():
            results_df.loc[k, ("train", key)] = val
        for key, val in auc_val.items():
            results_df.loc[k, ("val", key)] = val
        for key, val in auc_test.items():
            results_df.loc[k, ("test", key)] = val

    print(results_df)

    file.close()


if __name__ == '__main__':
    main()
