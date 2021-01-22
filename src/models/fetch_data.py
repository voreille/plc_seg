from pathlib import Path
import os

import tensorflow as tf
import pandas as pd
import numpy as np
import SimpleITK as sitk
import random
from tensorflow.python.ops.gen_string_ops import regex_full_match


def to_np(x):
    return np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0))


def bb_intersection(*args):
    pos_max = args[0][3:]
    pos_min = args[0][:3]
    for bb in args:
        pos_max = np.minimum(pos_max, bb[3:])
        pos_min = np.maximum(pos_min, bb[:3])

    return np.concatenate([pos_min, pos_max], axis=0)


def get_bb_mask(mask_sitk):
    mask = sitk.GetArrayFromImage(mask_sitk)
    positions = np.where(mask != 0)
    z_min = np.min(positions[0])
    y_min = np.min(positions[1])
    x_min = np.min(positions[2])
    z_max = np.max(positions[0])
    y_max = np.max(positions[1])
    x_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, z_max, y_max


def mask_center(mask_sitk):
    x_min, y_min, z_min, x_max, z_max, y_max = get_bb_mask(mask_sitk)
    center_ind = [
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2,
    ]
    return mask_sitk.TransformContinuousIndexToPhysicalPoint(center_ind)


def get_radius_mask(mask_sitk):
    x_min, y_min, z_min, x_max, z_max, y_max = get_bb_mask(mask_sitk)
    return (x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2


def bb_image(image):
    origin = image.GetOrigin()
    max_position = origin + np.array(
        [image.GetWidth(),
         image.GetHeight(),
         image.GetDepth()]) * image.GetSpacing()
    return np.concatenate([origin, max_position], axis=0)


def random_center(mask,
                  bb_image,
                  output_shape=(128, 128, 128),
                  spacing=1.0,
                  probability_include_mask=1.0):
    center_mask = mask_center(mask)
    max_radius = (output_shape[0] // 2) * spacing
    if np.random.random() < probability_include_mask:
        max_radius -= get_radius_mask(mask)
    else:
        max_radius += get_radius_mask(mask)

    radius = np.sqrt(np.random.uniform(high=max_radius**2))
    theta = np.random.uniform(high=np.pi)
    phi = np.random.uniform(high=2 * np.pi)
    delta_position = np.array([
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)
    ])

    center = center_mask + delta_position
    pos_max = center + (np.array(output_shape) // 2) * spacing
    pos_min = center - (np.array(output_shape) // 2) * spacing
    # Check if the the output image is outside the image and correct it
    delta_out = bb_image[3:] - pos_max
    delta_out[delta_out > 0] = 0
    center = center + delta_out

    delta_out = bb_image[:3] - pos_max
    delta_out[delta_out < 0] = 0
    center = center + delta_out

    return center


def get_ct_range(anat_str):
    wwl_dict = {
        "lung": [1500, -600],
        "brain": [80, 40],
        "head_neck_soft_tissue": [350, 60],
    }

    return (wwl_dict[anat_str][1] - wwl_dict[anat_str][0] * 0.5,
            wwl_dict[anat_str][1] + wwl_dict[anat_str][0] * 0.5)


def get_tf_dataset(
    patient_list,
    path_data_nii,
    path_mask_lung_nii,
    augment_angles=None,
    augment_mirror=False,
    num_parallel_calls=None,
    output_shape=(144, 144, 144),
    spacing=(1, 1, 1),
    ct_window_str="lung",
    return_patient_name=False,
    interp_order=3,
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
    f = lambda x, patient_name: (*tf_parse_image(x,
                                                 Path(path_data_nii),
                                                 Path(path_mask_lung_nii),
                                                 output_shape=output_shape,
                                                 spacing=spacing,
                                                 ct_window_str=ct_window_str,
                                                 augment_mirror=augment_mirror,
                                                 augment_angles=augment_angles,
                                                 interp_order=interp_order),
                                 patient_name)

    # Mapping the parsing function
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    image_ds = patient_folders_ds.map(f, num_parallel_calls=num_parallel_calls)
    if return_patient_name is False:
        image_ds = image_ds.map(lambda x, y, z: (x, y))
    return image_ds


def tf_parse_image(patient,
                   path_nii,
                   path_lung_mask_nii,
                   output_shape=(128, 128, 128),
                   spacing=(1, 1, 1),
                   ct_window_str="lung",
                   augment_mirror=True,
                   augment_angles=True,
                   interp_order=3):

    f = lambda x: parse_image(x,
                              path_nii,
                              path_lung_mask_nii,
                              output_shape=output_shape,
                              spacing=spacing,
                              ct_window_str=ct_window_str,
                              augment_mirror=augment_mirror,
                              augment_angles=augment_angles,
                              interp_order=interp_order)
    image, mask = tf.py_function(f,
                                 inp=[patient],
                                 Tout=(tf.float32, tf.float32))
    image.set_shape(output_shape + (2, ))
    mask.set_shape(output_shape + (1, ))

    return image, mask


def normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


def parse_image(patient,
                path_nii,
                path_lung_mask_nii,
                output_shape=(144, 144, 144),
                spacing=(1, 1, 1),
                ct_window_str="lung",
                augment_mirror=False,
                augment_angles=None,
                interp_order=3):
    """Parse the raw data of HECKTOR 2020

    Args:
        folder_name ([Path]): the path of the folder containing 
        the 3 sitk images (ct, pt and mask)
    """
    output_shape = np.array(output_shape)
    patient_name = patient.numpy().decode("utf-8")
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
    resampler = sitk.ResampleImageFilter()
    if interp_order == 3:
        resampler.SetInterpolator(sitk.sitkBSpline)
    # compute center
    bb_max = bb_intersection(bb_image(ct_sitk), bb_image(pt_sitk))
    center = random_center(mask_gtvt_sitk,
                           bb_max,
                           output_shape=output_shape,
                           spacing=spacing,
                           probability_include_mask=1.0)

    # Define origin
    origin = center - output_shape // 2

    resampler.SetOutputOrigin(tuple(*origin))
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(output_shape)

    if augment_angles:
        augment_angles = np.array(augment_angles) * np.pi / 180
        transform = sitk.Euler3DTransform()
        transform.SetCenter(np.squeeze(center))
        transform.SetRotation(*(
            np.random.random_sample(3) * 2 * augment_angles - augment_angles))
        resampler.SetTransform(transform)

    ct_sitk = resampler.Execute(ct_sitk)
    pt_sitk = resampler.Execute(pt_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_gtvt = to_np(resampler.Execute(mask_gtvt_sitk))
    mask_gtvl = to_np(resampler.Execute(mask_gtvl_sitk))
    mask_lung = to_np(resampler.Execute(mask_lung_sitk))

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)
    hu_low, hu_high = get_ct_range(ct_window_str)
    ct[ct > hu_high] = hu_high
    ct[ct < hu_low] = hu_low
    ct = (2 * ct - hu_high - hu_low) / (hu_high - hu_low)

    pt = normalize_image(pt)

    image = np.stack([ct, pt], axis=-1)
    mask = np.stack([mask_lung, mask_gtvt, mask_gtvl], axis=-1)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    if augment_mirror:
        if bool(random.getrandbits(1)):
            mask = np.flip(mask, axis=0)
            image = np.flip(image, axis=0)
    return image, mask
