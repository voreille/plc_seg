from pathlib import Path
import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np
import SimpleITK as sitk
import random
from tensorflow.python.ops.gen_string_ops import regex_full_match


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def bb_intersection(*args):
    pos_max = args[0][3:]
    pos_min = args[0][:3]
    for bb in args:
        pos_max = np.minimum(pos_max, bb[3:])
        pos_min = np.maximum(pos_min, bb[:3])

    return np.concatenate([pos_min, pos_max], axis=0)


def get_bb_mask_voxel(mask_sitk):
    mask = sitk.GetArrayFromImage(mask_sitk)
    positions = np.where(mask != 0)
    z_min = np.min(positions[0])
    y_min = np.min(positions[1])
    x_min = np.min(positions[2])
    z_max = np.max(positions[0])
    y_max = np.max(positions[1])
    x_max = np.max(positions[2])
    return x_min, y_min, z_min, x_max, z_max, y_max


def get_bb_mask_mm(mask_sitk):
    x_min, y_min, z_min, x_max, z_max, y_max = get_bb_mask_voxel(mask_sitk)
    return (*mask_sitk.TransformIndexToPhysicalPoint(
        [int(x_min), int(y_min), int(z_min)]),
            *mask_sitk.TransformIndexToPhysicalPoint(
                [int(x_max), int(y_max), int(z_max)]))


def mask_center(mask_sitk):
    x_min, y_min, z_min, x_max, z_max, y_max = get_bb_mask_voxel(mask_sitk)
    center_ind = [
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2,
    ]
    return np.array(
        mask_sitk.TransformContinuousIndexToPhysicalPoint(center_ind))


def get_radius_mask_2d(mask_sitk):
    x_min, y_min, z_min, x_max, z_max, y_max = get_bb_mask_mm(mask_sitk)
    return np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) * 0.5


def bb_image(image):
    origin = image.GetOrigin()
    max_position = origin + np.array(
        [image.GetWidth(),
         image.GetHeight(),
         image.GetDepth()]) * image.GetSpacing()
    return np.concatenate([origin, max_position], axis=0)


def get_center(
        mask,
        bb_image,
        output_shape=(256, 256, 1),
        spacing=(1, 1, 1),
):
    center = mask_center(mask)
    pos_max = center + (np.array(output_shape) // 2) * spacing
    pos_min = center - (np.array(output_shape) // 2) * spacing
    # Check if the the output image is outside the image and correct it
    delta_out = bb_image[3:] - pos_max
    delta_out[delta_out > 0] = 0
    center = center + delta_out

    delta_out = bb_image[:3] - pos_min
    delta_out[delta_out < 0] = 0
    center = center + delta_out

    return center


def get_random_center(mask,
                      bb_image,
                      output_shape=(256, 256, 1),
                      spacing=(1, 1, 1),
                      probability_include_mask=1.0):
    center_mask = mask_center(mask)
    bb_mask = np.array(get_bb_mask_mm(mask))
    max_radius = (output_shape[0] // 2) * spacing[0]
    if np.random.random() < probability_include_mask:
        max_radius -= np.min(bb_mask[3:5] - bb_mask[:2]) / 2
    else:
        max_radius += np.max(bb_mask[3:5] - bb_mask[:2])

    radius = np.sqrt(np.random.uniform(high=max_radius**2))
    # radius = max_radius
    theta = np.random.uniform(high=2 * np.pi)
    # delta_z = np.random.normal(scale=(bb_mask[-1] - bb_mask[2]) / 2 / 1.96)

    dz = 0.9 * (bb_mask[-1] - bb_mask[2]) / 2
    delta_z = np.random.uniform(low=-dz, high=dz)
    # delta_z = 0
    delta_position = np.array(
        [radius * np.cos(theta), radius * np.sin(theta), delta_z])

    center = center_mask + delta_position
    pos_max = center + (np.array(output_shape) // 2) * spacing
    pos_min = center - (np.array(output_shape) // 2) * spacing
    # Check if the the output image is outside the image and correct it
    delta_out = bb_image[3:] - pos_max
    delta_out[delta_out > 0] = 0
    center = center + delta_out

    delta_out = bb_image[:3] - pos_min
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
    f = lambda patient_name: (*tf_parse_image(
        patient_name,
        Path(path_data_nii),
        Path(path_mask_lung_nii),
        output_shape=output_shape,
        spacing=spacing,
        ct_window_str=ct_window_str,
        random_center=random_center,
        augment_mirror=augment_mirror,
        augment_angles=augment_angles,
        interp_order=interp_order,
        mask_smoothing=mask_smoothing,
        smoothing_radius=smoothing_radius,
    ), patient_name)

    # Mapping the parsing function
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    image_ds = patient_ds.map(f, num_parallel_calls=num_parallel_calls).cache()
    if return_patient_name is False:
        image_ds = image_ds.map(lambda x, y, z: (x, y))
    return image_ds


def tf_parse_image(
    patient,
    path_nii,
    path_lung_mask_nii,
    output_shape=(256, 256),
    spacing=(1, 1, 1),
    ct_window_str="lung",
    random_center=False,
    augment_mirror=True,
    augment_angles=True,
    interp_order=3,
    mask_smoothing=False,
    smoothing_radius=3,
):

    f = lambda x: parse_image(
        x,
        path_nii,
        path_lung_mask_nii,
        output_shape=output_shape,
        spacing=spacing,
        ct_window_str=ct_window_str,
        random_center=random_center,
        augment_mirror=augment_mirror,
        augment_angles=augment_angles,
        interp_order=interp_order,
        mask_smoothing=mask_smoothing,
        smoothing_radius=smoothing_radius,
    )
    image, mask = tf.py_function(f,
                                 inp=[patient],
                                 Tout=(tf.float32, tf.float32))
    image.set_shape(output_shape + (3, ))
    mask.set_shape(output_shape + (3, ))

    return image, mask


def normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


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


def parse_image(
    patient,
    path_nii,
    path_lung_mask_nii,
    output_shape=(256, 256),
    spacing=(1, 1, 1),
    ct_window_str="lung",
    random_center=False,
    augment_mirror=False,
    augment_angles=None,
    interp_order=3,
    mask_smoothing=False,
    smoothing_radius=3,
):
    """Parse the raw data of HECKTOR 2020

    Args:
        folder_name ([Path]): the path of the folder containing 
        the 3 sitk images (ct, pt and mask)
    """
    if len(output_shape) == 2:
        output_shape = tuple(output_shape) + (1, )
    output_shape = np.array(output_shape)
    patient_name = patient.numpy().decode("utf-8")
    t1 = time.time()
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
    print(f"Time reading the files for patient {patient} : {time.time()-t1}")
    t1 = time.time()
    resampler = sitk.ResampleImageFilter()
    if interp_order == 3:
        resampler.SetInterpolator(sitk.sitkBSpline)
    # compute center
    bb_max = bb_intersection(bb_image(ct_sitk), bb_image(pt_sitk))
    if random_center:
        center = get_random_center(mask_gtvt_sitk,
                                   bb_max,
                                   output_shape=output_shape,
                                   spacing=spacing,
                                   probability_include_mask=1.0)
    else:
        center = get_center(mask_gtvt_sitk,
                            bb_max,
                            output_shape=output_shape,
                            spacing=spacing)

    # Define origin
    origin = center - output_shape // 2

    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(tuple([int(k) for k in output_shape]))

    if augment_angles:
        augment_angles = np.array(augment_angles) * np.pi / 180
        transform = sitk.Euler3DTransform()
        transform.SetCenter(mask_center(mask_gtvt_sitk))
        angles = np.random.random_sample(
            3) * 2 * augment_angles - augment_angles
        transform.SetRotation(*angles)
        # transform.SetRotation(*augment_angles) # debugging purpose
        resampler.SetTransform(transform)

    ct_sitk = resampler.Execute(ct_sitk)
    pt_sitk = resampler.Execute(pt_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_gtvt_sitk = resampler.Execute(mask_gtvt_sitk)
    mask_gtvl_sitk = resampler.Execute(mask_gtvl_sitk)
    mask_lung_sitk = resampler.Execute(mask_lung_sitk)
    mask_lung1_sitk, mask_lung2_sitk = split_lung_mask(mask_lung_sitk)
    if mask_smoothing:
        smoother = sitk.BinaryMedianImageFilter()
        smoother.SetRadius(int(smoothing_radius))
        mask_gtvt_sitk = smoother.Execute(mask_gtvt_sitk)
        mask_gtvl_sitk = smoother.Execute(mask_gtvl_sitk)
        mask_lung1_sitk = smoother.Execute(mask_lung1_sitk)
        mask_lung2_sitk = smoother.Execute(mask_lung2_sitk)
    mask_gtvt = to_np(mask_gtvt_sitk)
    mask_gtvl = to_np(mask_gtvl_sitk)
    mask_lung1 = to_np(mask_lung1_sitk)
    mask_lung2 = to_np(mask_lung2_sitk)

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)
    hu_low, hu_high = get_ct_range(ct_window_str)
    ct[ct > hu_high] = hu_high
    ct[ct < hu_low] = hu_low
    ct = (2 * ct - hu_high - hu_low) / (hu_high - hu_low)

    pt = normalize_image(pt)

    image = np.stack([ct, pt, np.zeros_like(ct)], axis=-1)
    mask = np.stack([mask_gtvt, mask_gtvl, mask_lung1 + mask_lung2], axis=-1)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    if augment_mirror:
        if bool(random.getrandbits(1)):
            mask = np.flip(mask, axis=0)
            image = np.flip(image, axis=0)

    print(f"Time preprocessing for patient {patient} : {time.time()-t1}")
    return image, mask
