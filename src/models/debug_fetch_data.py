from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.fetch_data_2d_fixed import (get_tf_dataset, tf_parse_image,
                                            parse_image, slice_image)

path_data_nii = Path("/home/val/python_wkspce/plc_seg/data/interim/nii_raw")
path_mask_lung_nii = Path(
    "/home/val/python_wkspce/plc_seg/data/interim/lung_contours")

patient_list = [
    f.name.split("__")[0] for f in path_mask_lung_nii.rglob("*LUNG*")
]

patient_list = patient_list[:1]
patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)

sitk_volume_ds = patient_ds.map(lambda patient_name: (*tf_parse_image(
    patient_name,
    Path(path_data_nii),
    Path(path_mask_lung_nii),
), )).as_numpy_iterator()

f = lambda inp: slice_image(inp,
                            augment_angles=(0, 0, 0),
                            output_shape=(256, 256),
                            spacing=(1, 1, 1),
                            random_center=False,
                            interp_order_image=3,
                            interp_order_mask=0,
                            fill_mode='constant',
                            fill_value=0.0)

for p in sitk_volume_ds:
    image, mask = f(p)