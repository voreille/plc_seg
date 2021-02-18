from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.fetch_data_2d import get_tf_dataset, tf_parse_image, parse_image

path_data_nii = Path("/home/val/python_wkspce/plc_seg/data/interim/nii_raw")
path_mask_lung_nii = Path(
    "/home/val/python_wkspce/plc_seg/data/interim/lung_contours")

patient_list = [
    f.name.split("__")[0] for f in path_mask_lung_nii.rglob("*LUNG*")
]
patient_list = patient_list
patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)

for p in patient_ds:
    image, mask = parse_image(
        p,
        path_data_nii,
        path_mask_lung_nii,
        augment_angles=(20, 20, 90),
        random_center=True,
    )
