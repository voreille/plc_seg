import os
from pathlib import Path

import dotenv
import tensorflow as tf
import h5py
import pandas as pd
from tensorflow.python.training.tracking.util import CheckpointLoadStatus

from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import unet_model, CustomModel
from src.models.losses_2d import dice_coe_1_hard
from src.models.model_evaluation import compute_plc_volume, evaluate_model

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))

path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])
# model = "model_0-75_alpha_1-00_wplc_1-00_wt_1-00_wl_40-00_splc_early_stop"
# model = "model_0-75_alpha_1-00_wplc_0-00_wt_0-00_wl_40-00_splc_early_stop"
model = "model_0-75_alpha_1-00_wplc_1-00_wt_1-00_wl_40-00_splc_early_stop"

checkpoint = False
if checkpoint:
    model_path = project_dir / f"models/{model}/checkpoint/checkpoint"
else:
    # model_path = f"/home/val/python_wkspce/plc_seg/models/{model}/epochs_50"
    model_path = f"/home/val/python_wkspce/plc_seg/models/early_stop/{model}/final"

output_directory = project_dir / "data/plc_volume"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / f"{model}.csv"

bs = 4
image_size = (256, 256)


def main():
    file_train = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d_pet_normalized/train.hdf5",
        "r")
    file_test = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d_pet_normalized/test.hdf5",
        "r")
    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")

    if checkpoint:
        model_ = unet_model(3, input_shape=image_size + (3, ))
        model = CustomModel(model_.inputs, model_.outputs)
    else:
        model = tf.keras.models.load_model(model_path, compile=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
    )
    if checkpoint:
        model.load_weights(model_path)

    data_test = get_tf_data(
        file_test,
        clinical_df,
        output_shape_image=(256, 256),
        random_slice=False,
        centered_on_gtvt=True,
    ).cache().batch(2)

    results = evaluate_model(
        model,
        data_test,
        str_df="test",
        w_gtvl=4,
        w_gtvt=1,
        w_lung=1,
        alpha=0.75,
        s_gtvl=40,
    )

    print(f" REstusl : {results}")

    file_train.close()
    file_test.close()


if __name__ == '__main__':
    main()