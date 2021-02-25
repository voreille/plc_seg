import os
from pathlib import Path

import dotenv
import tensorflow as tf
import h5py
import pandas as pd

from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import unet_model, CustomModel

project_dir = Path(__file__).resolve().parents[2]
dotenv_path = project_dir / ".env"
dotenv.load_dotenv(str(dotenv_path))

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

model_path = project_dir / "models/first_model"
checkpoint_filepath = project_dir / "models/checkpoint"

bs = 4
n_epochs = 1000
n_prefetch = 20
image_size = (224, 224)


def main():
    file_train = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d/train.hdf5", "r")
    file_test = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d/test.hdf5", "r")
    clinical_df = pd.read_excel(
        "/home/val/python_wkspce/plc_seg/data/List_lymphangite_radiomics_SNM2020_MJ.xlsx"
    ).set_index("patient_id")
    data_test = get_tf_data(
        file_test,
        clinical_df,
        output_shape=(224, 224),
        random_slice=False,
    ).cache().batch(bs)
    data_train = get_tf_data(
        file_train,
        clinical_df,
        output_shape=(224, 224),
        random_slice=True,
        n_repeat=10,
        num_parallel_calls='auto',
    ).batch(bs)

    model_ = unet_model(3, input_shape=image_size + (3, ))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = CustomModel(model_.inputs, model_.outputs)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False,
        # save_freq=1,
    )

    # We don't passs a loss or metrics here.
    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
    )

    model.fit(
        x=data_train,
        epochs=n_epochs,
        validation_data=data_test,
        callbacks=[model_checkpoint_callback],
    )
    model.save(model_path)
    file_train.close()
    file_test.close()


if __name__ == '__main__':
    main()