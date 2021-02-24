from pathlib import Path
import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np
import SimpleITK as sitk
import random
from tensorflow.python.ops.gen_string_ops import regex_full_match

from src.models.data_augmentation import compute_coordinate_matrix, slice_image
from src.utils.image import get_ct_range


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def get_tf_dataset(
    patient_list,
    path_data_nii,
    path_mask_lung_nii,
    random_center=False,
    augment_angles=None,
    augment_mirror=False,
    num_parallel_calls=None,
    output_shape=(256, 256),
    spacing=(1, 1, 1),
    ct_window_str="lung",
    return_patient_name=False,
    interp_order=3,
    mask_smoothing=False,
    smoothing_radius=3,
    center_on_which_mask=0,
    interp_order_image=3,
    interp_order_mask=0,
    fill_mode="constant",
    fill_value=0.0,
):
    """

    Args:
        path_data_nii (str): Path to the folder containing the original nii 
        bbox_path (str): the path to the bounding box
        augment_shift (float, optional): magnitute of random shifts. Defaults to None.
        augment_mirror (bool, optional): Whether to apply random sagittal mirroring. Defaults to False.
        augment_angles (tuple, optional): whether to apply random rotation, of maximal amplitude of
                                         (angle_x, angle_y, angle_z) where the angle are defined in degree.
        num_parallel_calls (int, optional): num of CPU for reading the data. If None tensorfow decides.
        regex (str, optional): Regex expression to filter patient names. Defaults to None.
                                If None no filtering is applied.
        regex_in (bool, optional): Wether to exclude or include the regex. Defaults to True.
        output_shape (tuple, optional): Output shape of the spatial domain. Defaults to (144,144,144).
        return_patient_name (bool, optional): Wether to return the patient name


    Returns:
        [type]: [description]
    """

    # Generate a tf.dataset for the paths to the folder
    patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)

    # get the tf.dataset to return the path and the name of the patient

    # Define the mapping to parse the paths
    def f1(patient_name):
        return tf_parse_image(
            patient_name,
            Path(path_data_nii),
            Path(path_mask_lung_nii),
            ct_window_str=ct_window_str,
        )

    def f2(
        ct,
        pt,
        mask_lung1,
        mask_lung2,
        mask_gtvt,
        mask_gtvl,
        coordinate_matrix_ct,
        coordinate_matrix_pt,
        coordinate_matrix_mask_lung,
    ):
        return tf_slice_image(
            ct,
            pt,
            mask_lung1,
            mask_lung2,
            mask_gtvt,
            mask_gtvl,
            coordinate_matrix_ct,
            coordinate_matrix_pt,
            coordinate_matrix_mask_lung,
            augment_angles=augment_angles,
            output_shape=output_shape,
            spacing=spacing,
            random_center=random_center,
            interp_order_image=interp_order_image,
            interp_order_mask=interp_order_mask,
            fill_mode=fill_mode,
            fill_value=fill_value,
        )

    # Mapping the parsing function
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    volume_ds = patient_ds.map(f1, num_parallel_calls=num_parallel_calls).cache()
    image_ds = volume_ds.map(f2, num_parallel_calls=num_parallel_calls)

    # if return_patient_name is False:
    #     image_ds = image_ds.map(lambda x, y, z: (x, y))
    return image_ds


def tf_slice_image(
    ct,
    pt,
    mask_lung1,
    mask_lung2,
    mask_gtvt,
    mask_gtvl,
    coordinate_matrix_ct,
    coordinate_matrix_pt,
    coordinate_matrix_mask_lung,
    augment_angles=(0, 0, 0),
    output_shape=(256, 256),
    spacing=(1, 1, 1),
    random_center=False,
    interp_order_image=3,
    interp_order_mask=0,
    fill_mode='constant',
    fill_value=0.0,
):
    def f(ct, pt, mask_lung1, mask_lung2, mask_gtvt, mask_gtvl,
          coordinate_matrix_ct, coordinate_matrix_pt,
          coordinate_matrix_mask_lung):
        return slice_image(
            ct,
            pt,
            mask_lung1,
            mask_lung2,
            mask_gtvt,
            mask_gtvl,
            coordinate_matrix_ct,
            coordinate_matrix_pt,
            coordinate_matrix_mask_lung,
            augment_angles=augment_angles,
            output_shape=output_shape,
            spacing=spacing,
            random_center=random_center,
            interp_order_image=interp_order_image,
            interp_order_mask=interp_order_mask,
            fill_mode=fill_mode,
            fill_value=fill_value,
        )

    i, m = tf.py_function(
        f,
        inp=[
            ct, pt, mask_lung1, mask_lung2, mask_gtvt, mask_gtvl,
            coordinate_matrix_ct, coordinate_matrix_pt,
            coordinate_matrix_mask_lung
        ],
        Tout=(
            tf.float32,
            tf.float32,
        ),
    )

    i.set_shape(output_shape + (3, ))
    m.set_shape(output_shape + (3, ))
    return i, m


def tf_parse_image(
    patient,
    path_nii,
    path_lung_mask_nii,
    ct_window_str="lung",
):
    def f(p):
        return parse_image(
            p,
            path_nii,
            path_lung_mask_nii,
            ct_window_str=ct_window_str,
        )

    (ct, pt, mask_lung1, mask_lung2, mask_gtvt, mask_gtvl,
     coordinate_matrix_ct, coordinate_matrix_pt,
     coordinate_matrix_mask_lung) = tf.py_function(f,
                                                   inp=[patient],
                                                   Tout=(
                                                       tf.float32,
                                                       tf.float32,
                                                       tf.uint8,
                                                       tf.uint8,
                                                       tf.uint8,
                                                       tf.uint8,
                                                       tf.float32,
                                                       tf.float32,
                                                       tf.float32,
                                                   ))

    # image.set_shape(output_shape + (2, ))
    # mask.set_shape(output_shape + (4, ))
    return (ct, pt, mask_lung1, mask_lung2, mask_gtvt, mask_gtvl,
            coordinate_matrix_ct, coordinate_matrix_pt,
            coordinate_matrix_mask_lung)


def parse_image(
    patient,
    path_nii,
    path_lung_mask_nii,
    ct_window_str="lung",
):
    """Parse the raw data of HECKTOR 2020

    Args:
        folder_name ([Path]): the path of the folder containing 
        the 3 sitk images (ct, pt and mask)
    """
    patient_name = patient.numpy().decode("utf-8")
    # patient_name = patient
    ct_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__CT.nii.gz")).resolve()))
    pt_sitk = sitk.ReadImage(
        str((path_nii / (patient_name + "__PT.nii.gz")).resolve()))
    mask_gtvt_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_T__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_gtvl_sitk = sitk.ReadImage(
        str((path_nii /
             (patient_name + "__GTV_L__RTSTRUCT__CT.nii.gz")).resolve()))
    mask_lung_sitk = sitk.ReadImage(
        str((path_lung_mask_nii /
             (patient_name + "__LUNG__SEG__CT.nii.gz")).resolve()))

    mask_lung1_sitk, mask_lung2_sitk = split_lung_mask(mask_lung_sitk)
    mask_gtvt = to_np(mask_gtvt_sitk).astype(np.uint8)
    mask_gtvl = to_np(mask_gtvl_sitk).astype(np.uint8)
    mask_lung1 = to_np(mask_lung1_sitk).astype(np.uint8)
    mask_lung2 = to_np(mask_lung2_sitk).astype(np.uint8)

    coordinate_matrix_ct = compute_coordinate_matrix(
        origin=ct_sitk.GetOrigin(),
        pixel_spacing=ct_sitk.GetSpacing(),
        direction=ct_sitk.GetDirection())

    coordinate_matrix_pt = compute_coordinate_matrix(
        origin=pt_sitk.GetOrigin(),
        pixel_spacing=pt_sitk.GetSpacing(),
        direction=pt_sitk.GetDirection())

    coordinate_matrix_mask_lung = compute_coordinate_matrix(
        origin=mask_lung_sitk.GetOrigin(),
        pixel_spacing=mask_lung_sitk.GetSpacing(),
        direction=mask_lung_sitk.GetDirection())

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)

    hu_low, hu_high = get_ct_range(ct_window_str)
    ct[ct > hu_high] = hu_high
    ct[ct < hu_low] = hu_low
    ct = (2 * ct - hu_high - hu_low) / (hu_high - hu_low)

    # pt = normalize_image(pt)

    # mask = np.stack([mask_gtvt, mask_gtvl, mask_lung1, mask_lung2], axis=-1)
    # mask[mask >= 0.5] = 1
    # mask[mask < 0.5] = 0

    return (ct, pt, mask_lung1, mask_lung2, mask_gtvt, mask_gtvl,
            coordinate_matrix_ct, coordinate_matrix_pt,
            coordinate_matrix_mask_lung)


def split_lung_mask(lung_sitk):
    lung = sitk.GetArrayFromImage(lung_sitk)
    lung1 = np.zeros_like(lung, dtype=int)
    lung2 = np.zeros_like(lung, dtype=int)
    lung1[lung == 1] = 1
    lung2[lung == 2] = 1
    lung1 = sitk.GetImageFromArray(lung1)
    lung1.SetOrigin(lung_sitk.GetOrigin())
    lung1.SetSpacing(lung_sitk.GetSpacing())
    lung1.SetDirection(lung_sitk.GetDirection())
    lung2 = sitk.GetImageFromArray(lung2)
    lung2.SetOrigin(lung_sitk.GetOrigin())
    lung2.SetSpacing(lung_sitk.GetSpacing())
    lung2.SetDirection(lung_sitk.GetDirection())
    return lung1, lung2
