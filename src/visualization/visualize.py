import os
from pathlib import Path

import dotenv
import tensorflow as tf
import h5py
import pandas as pd
import matplotlib.pyplot as plt

from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.train_model_2d import CustomLoss

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))

path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

model = "model1"
model_path = project_dir / f"models/clean_model/{model}"

output_directory = project_dir / "reports/visualizations/model1"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / f"{model}.csv"

bs = 4
image_size = (256, 256)


def main():
    # file_train = h5py.File(
    #     "/home/val/python_wkspce/plc_seg/data/processed/2d_pet_normalized/train.hdf5",
    #     "r")
    file_test = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d_pet_normalized/test.hdf5",
        "r")
    clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")

    model = tf.keras.models.load_model(model_path, compile=False)

    data_test = get_tf_data(
        file_test,
        clinical_df,
        output_shape_image=(256, 256),
        random_slice=False,
        centered_on_gtvt=True,
        return_complete_gtvl=True,
        return_patient=True,
        return_plc_status=False,
    ).batch(4)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        loss=CustomLoss(),
        optimizer=optimizer,
        run_eagerly=False,
    )

    for x, y_true, patient_id in data_test.take(1).as_numpy_iterator():
        y_pred = model(x).numpy()
        for i in range(x.shape[0]):
            print(y_pred.shape)

    # file_train.close()
    file_test.close()


if __name__ == '__main__':
    main()