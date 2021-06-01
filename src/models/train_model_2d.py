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

from src.models.losses_2d import CustomLoss, gtvl_loss, dice_coe_1_hard
from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import unet_model, Unet, UnetIantsen
from src.models.model_evaluation import compute_plc_volume, compute_results
from src.models.utils import plot_fig

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))
log_dir = project_dir / ("logs/fit/" +
                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

split_path = project_dir / "data/split.csv"

# model_pretrained = "model_seg_gtvt_iantsen"
model_pretrained = "model_seg_gtvt_iantsen"
model = "model_iantsen_gtvl_pretrained_on_gtvt_lung"
# model = "model_seg_gtvt_iantsen"
path_model_pretrained = project_dir / f"models/clean_model/{model_pretrained}"
path_model = project_dir / f"models/clean_model/{model}"

output_directory = project_dir / "data/plc_volume"
output_directory.mkdir(parents=True, exist_ok=True)
path_pred_volume = output_directory / f"{model}.csv"

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])
path_mean_sd = "/home/val/python_wkspce/plc_seg/data/mean_std.csv"

bs = 4
n_epochs = 100
n_prefetch = 20
image_size = (256, 256)

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)
tf.random.set_seed(2)

IMAGE_SHAPE = (256, 256, 3)
PRETRAIN = True


def model_builder(
    lr=1e-3,
    alpha=0.5,
    w_gtvl=1.0,
    w_gtvt=0.0,
    w_lung=0.0,
    model=None,
    model_str="mobilenet",
    run_eagerly=False,
):
    model_dict = {
        "iantsen": UnetIantsen,
        "mobilenet": Unet,
    }

    if model is None:
        model = model_dict[model_str](3, input_shape=IMAGE_SHAPE)
    else:
        if isinstance(model, Path):
            model = tf.keras.models.load_model(model, compile=False)
        for i in range(len(model.up_stack)):
            model.up_stack[i].trainable = False
        for i in range(len(model.down_stack)):
            model.down_stack[i].trainable = False

    def gtvl_metric(y_true, y_pred):
        return gtvl_loss(y_true, y_pred, pos_weight=alpha)

    def dice_gtvt(y_true, y_pred):
        return dice_coe_1_hard(y_true[..., 0], y_pred[..., 0])

    def dice_lung(y_true, y_pred):
        return dice_coe_1_hard(y_true[..., 2], y_pred[..., 2])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=CustomLoss(
            pos_weight=alpha,
            w_lung=w_lung,
            w_gtvt=w_gtvt,
            w_gtvl=w_gtvl,
        ),
        metrics=[gtvl_metric, dice_gtvt, dice_lung],
        run_eagerly=run_eagerly,
    )

    return model


def get_trainval_patient_list(df_path):
    df = pd.read_csv(df_path).set_index("patient_id")
    id_train = df[df["train_val_test"] == 0].index
    id_val = df[df["train_val_test"] == 1].index
    id_test = df[df["train_val_test"] == 2].index

    patient_list_test = [f"PatientLC_{i}" for i in id_test]
    patient_list_val = [f"PatientLC_{i}" for i in id_val]
    patient_list_train = [f"PatientLC_{i}" for i in id_train]

    return patient_list_train, patient_list_val, patient_list_test, df


class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, *args, min_epochs=0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.min_epochs:
            return
        else:
            super().on_epoch_end(epoch, logs=logs)


def main():
    h5_file = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/hdf5_2d/dataset.hdf5",
        "r")
    (patient_list_train, patient_list_val, patient_list_test,
     clinical_df) = get_trainval_patient_list(split_path)
    mean_sd_df = pd.read_csv(path_mean_sd).set_index("patient_id")
    clinical_df = pd.concat([clinical_df, mean_sd_df], axis=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    if PRETRAIN:
        data_train = get_tf_data(h5_file,
                                 clinical_df,
                                 output_shape_image=IMAGE_SHAPE,
                                 random_slice=True,
                                 label_to_contain="GTV T",
                                 random_shift=0,
                                 n_repeat=10,
                                 num_parallel_calls='auto',
                                 oversample_plc_neg=True,
                                 patient_list=patient_list_train).batch(bs)
        data_val = get_tf_data(
            h5_file,
            clinical_df,
            output_shape_image=IMAGE_SHAPE,
            random_slice=False,
            label_to_contain="GTV T",
            patient_list=patient_list_val,
        ).cache().batch(2)

        early_stop_callback = CustomEarlyStopping(
            min_epochs=0,
            monitor='val_dice_gtvt',
            # monitor='val_gtvl_metric',
            min_delta=0,
            patience=5,
            verbose=0,
            mode='max',
            restore_best_weights=True)
        model = model_builder(
            alpha=0.75,
            w_gtvl=0.0,
            w_gtvt=1.0,
            w_lung=1.0,
        )
        model.fit(
            x=data_train,
            epochs=n_epochs,
            validation_data=data_val,
            callbacks=[early_stop_callback, tensorboard_callback],
        )

        model.save(path_model_pretrained)
    else:
        model = path_model_pretrained

    data_train = get_tf_data(h5_file,
                             clinical_df,
                             output_shape_image=IMAGE_SHAPE,
                             random_slice=True,
                             label_to_contain="GTV L",
                             random_shift=0,
                             n_repeat=10,
                             num_parallel_calls='auto',
                             oversample_plc_neg=True,
                             patient_list=patient_list_train).batch(bs)
    data_val = get_tf_data(
        h5_file,
        clinical_df,
        output_shape_image=IMAGE_SHAPE,
        random_slice=False,
        label_to_contain="GTV L",
        patient_list=patient_list_val,
    ).cache().batch(2)

    early_stop_callback = CustomEarlyStopping(
        min_epochs=0,
        monitor='val_gtvl_metric',
        min_delta=0,
        patience=5,
        verbose=0,
        mode='min',
        restore_best_weights=True,
    )
    model = model_builder(
        alpha=0.25,
        w_gtvl=40.0,
        w_gtvt=0.0,
        w_lung=0.0,
        model=model,
        run_eagerly=False,
    )
    model.fit(
        x=data_train,
        epochs=n_epochs,
        validation_data=data_val,
        callbacks=[early_stop_callback, tensorboard_callback],
    )

    print(f"list of patient for testing : {patient_list_test}")
    model.save(path_model)
    volume_pred = compute_plc_volume(
        model,
        h5_file,
        clinical_df,
        image_shape=IMAGE_SHAPE,
    )
    volume_pred.to_csv(path_pred_volume)
    results = compute_results(volume_pred, clinical_df)
    print(f"The resulting AUCs are : {results}")
    h5_file.close()


if __name__ == '__main__':
    main()
