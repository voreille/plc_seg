from pathlib import Path
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.fetch_data_2d_fixed import (get_tf_dataset, tf_parse_image,
                                            parse_image, slice_image, tf_compute_mask, tf_plc)


# %%
path_data_nii = Path("/home/val/python_wkspce/plc_seg/data/interim/nii_raw")
path_mask_lung_nii = Path(
    "/home/val/python_wkspce/plc_seg/data/interim/lung_contours")

patient_list = [
    f.name.split("__")[0] for f in path_mask_lung_nii.rglob("*LUNG*")
]

patient_list = patient_list[:1]
patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)


# %%
path_clinical_info = "/home/val/python_wkspce/plc_seg/data/clinical_info.csv"
clinical_df = pd.read_csv(path_clinical_info).set_index("patient_id")


# %%
ds, volume_ds = get_tf_dataset(
    patient_list,
    path_data_nii,
    path_mask_lung_nii,
    clinical_df,
    random_center=False,
    augment_angles=(0, 0, 0),
    num_parallel_calls=None,
    output_shape=(256, 256),
    spacing=(1.5, 1.5, 1.5),
    ct_window_str="lung",
    interp_order_image=3,
    interp_order_mask=0,
    fill_mode="constant",
    fill_value=0.0,
)


# %%
volume_ds = volume_ds.repeat().take(1)


# %%
for vols in volume_ds.as_numpy_iterator():
    slice_image(*vols)


# %%
plt.figure(figsize=(4, 4))
plt.imshow(image[:,:,0],cmap='gray')
plt.imshow(mask[ :,:,3], cmap='jet', alpha=0.5)
plt.colorbar()


# %%



