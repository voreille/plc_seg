import os
from pathlib import Path

import dotenv
import tensorflow as tf
import h5py

from src.models.fetch_data_from_hdf5 import get_tf_data
from src.models.models_2d import unet_model
from src.models.losses_2d import (dice_coe_loss, dice_coe_hard,
                                  dice_coe_loss_1, dice_coe_hard_1)

path_data_nii = Path(os.environ["NII_PATH"])
path_mask_lung_nii = Path(os.environ["NII_LUNG_PATH"])
path_clinical_info = Path(os.environ["CLINIC_INFO_PATH"])

project_dir = Path(__file__).resolve().parents[2]
model_path = project_dir / "model/first_model"

bs = 4
n_epochs = 10
n_prefetch = 20
image_size = (224, 224)


def get_mask(y_true):
    if tf.reduce_sum(y_true[..., 2] * y_true[..., 1]) != 0:
        sick_lung = y_true[..., 2]
    else:
        sick_lung = y_true[..., 3]
    mask = sick_lung + y_true[..., 0] + y_true[..., 2]
    return mask


def loss(y_true, y_pred):
    """ 0: gtvt, 1: gtvl, 2: lung1, 3: lung2

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
    """
    mask = tf.map_fn(fn=get_mask, elems=y_true)
    mask = tf.where(mask > 0, x=1.0, y=0.0)
    # print(f"hey its the shape of the mask {mask.shape}")

    lung = y_true[..., 2] + y_true[..., 3]

    return (dice_coe_loss_1(y_true[..., 0], y_pred[..., 0]) +
            dice_coe_loss_1(lung, y_pred[..., 2]) +
            dice_coe_loss_1(mask * y_true[..., 1], mask * y_pred[..., 1]))


def main():
    file_train = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d/train.hdf5", "r")
    file_test = h5py.File(
        "/home/val/python_wkspce/plc_seg/data/processed/2d/test.hdf5", "r")
    data_test = get_tf_data(
        file_test,
        output_shape=(224, 224),
        random_slice=False,
    ).batch(bs)
    data_train = get_tf_data(
        file_train,
        output_shape=(224, 224),
        random_slice=True,
    ).batch(bs)

    model = unet_model(3, input_shape=image_size + (3, ))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        # metrics=dice_coe_hard,
    )

    model.fit(x=data_train, epochs=n_epochs, validation_data=data_test)
    model.save(model_path)
    file_train.close()
    file_test.close()


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]

    dotenv_path = project_dir / ".env"
    dotenv.load_dotenv(str(dotenv_path))
    main()