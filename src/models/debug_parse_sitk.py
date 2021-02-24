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


def f(patient):
    return parse_image(patient, path_data_nii, path_mask_lung_nii)


for p in patient_list:
    result = f(p)