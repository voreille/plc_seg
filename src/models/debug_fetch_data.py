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
patient_list = patient_list[:4]

patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)

n = 2
bs = 2
sradius = 3

data = get_tf_dataset(
    patient_list,
    path_data_nii,
    path_mask_lung_nii,
    random_center=True,
    augment_angles=(20, 20, 45),
    augment_mirror=False,
    num_parallel_calls=None,
    output_shape=(256, 256),
    spacing=(1, 1, 1),
    ct_window_str="lung",
    return_patient_name=True,
    interp_order=3,
    mask_smoothing=True,
    smoothing_radius=sradius,
)

X = data.batch(bs).take(n).as_numpy_iterator()

t1 = time.time()
for image, mask, patient in X:
    print(f"patient {patient} with an image shape {image.shape}")

elapsed = time.time() - t1
print(f"He ouai mon gars ça a mis tout ça de temps {elapsed}")
print(f"donc en moyenne ça fait {elapsed/n}")