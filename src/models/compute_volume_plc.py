import os
from pathlib import Path

import dotenv
import tensorflow as tf
import h5py
import pandas as pd

from src.models.model_evaluation import compute_plc_volume
from src.models.losses_2d import CustomLoss

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))

path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])
# model = "model_0-75_alpha_1-00_wplc_1-00_wt_1-00_wl_40-00_splc_early_stop"
# model = "model_0-75_alpha_1-00_wplc_0-00_wt_0-00_wl_40-00_splc_early_stop"

model = "model_gtvl_gtvt"
model_path = project_dir / f"models/clean_model/{model}"

output_directory = project_dir / "data/plc_volume"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / f"{model}.csv"

bs = 4
image_size = (256, 256)


def main():
    file_train = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed"
        "/2d_pet_normalized/train.hdf5", "r")
    file_test = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed"
        "/2d_pet_normalized/test.hdf5", "r")
    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")

    model = tf.keras.models.load_model(model_path, compile=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        loss=CustomLoss(),
        optimizer=optimizer,
        run_eagerly=False,
    )

    result = compute_plc_volume(model,
                                file_test,
                                clinical_df,
                                image_shape=image_size,
                                batch_size=bs)

    result.to_csv(str(output_path.resolve()))

    file_train.close()
    file_test.close()


if __name__ == '__main__':
    main()
