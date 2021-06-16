from pathlib import Path

import tensorflow as tf
import numpy as np
import SimpleITK as sitk

from src.models.data_augmentation import compute_coordinate_matrix, slice_image
from src.utils.image import get_ct_range


def to_np(x):
    return np.squeeze(np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0)))


def get_tf_dataset(
    patient_list,
    path_data_nii,
    path_mask_lung_nii,
    clinical_df,
    random_center=False,
    augment_angles=None,
    num_parallel_calls=None,
    output_shape=(256, 256),
    spacing=(1, 1, 1),
    ct_window_str="lung",
    interp_order_image=3,
    interp_order_mask=0,
    fill_mode="constant",
    fill_value=0.0,
):

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
    # if num_parallel_calls is None:
    #     num_parallel_calls = tf.data.experimental.AUTOTUNE
    num_parallel_calls = tf.data.experimental.AUTOTUNE

    volume_ds = patient_ds.map(f1,
                               num_parallel_calls=num_parallel_calls).cache()
    image_ds = volume_ds.map(f2, num_parallel_calls=num_parallel_calls)

    # if return_patient_name is False:
    #     image_ds = image_ds.map(lambda x, y, z: (x, y))
    plc_ds = patient_ds.map(lambda p: tf_plc(p, clinical_df))
    image_ds = tf.data.Dataset.zip(
        (image_ds, plc_ds)).map(tf_compute_mask,
                                num_parallel_calls=num_parallel_calls)
    return image_ds


def tf_compute_mask(image_ds_output, plc_ds_output):
    image, mask = image_ds_output
    plc_status, sick_lung_axis = plc_ds_output
    mask_loss = tf.where(plc_status == 1,
                         x=1 - mask[..., sick_lung_axis[0]] + mask[..., 0] +
                         mask[..., 1],
                         y=tf.ones(mask.shape[:2]))
    mask_gtvl = tf.where(plc_status == 1,
                         x=mask[..., 1],
                         y=tf.zeros(mask.shape[:2]))

    mask_loss = tf.where(mask_loss >= 1, x=1.0, y=0.0)
    return image, tf.stack(
        [mask[..., 0], mask_gtvl, mask[..., 2] + mask[..., 3], mask_loss],
        axis=-1)


def tf_plc(patient, clinical_df):
    def parse_plc(patient):
        patient_id = int(patient.numpy().decode("utf-8").split('_')[-1])
        plc_status = int(clinical_df.loc[patient_id, "plc_status"])
        sick_lung_axis = int(clinical_df.loc[patient_id, "sick_lung_axis"])
        return np.array([plc_status]), np.array([sick_lung_axis])

    plc_status, sick_lung_axis = tf.py_function(
        parse_plc,
        inp=[patient],
        Tout=(tf.int8, tf.int32),
    )
    plc_status.set_shape((1, ))
    sick_lung_axis.set_shape((1, ))
    return plc_status, sick_lung_axis


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
    m.set_shape(output_shape + (4, ))
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
    ct.set_shape((None, None, None))
    pt.set_shape((None, None, None))
    mask_lung1.set_shape((None, None, None))
    mask_lung2.set_shape((None, None, None))
    mask_gtvt.set_shape((None, None, None))
    mask_gtvl.set_shape((None, None, None))
    coordinate_matrix_ct.set_shape((4, 4))
    coordinate_matrix_pt.set_shape((4, 4))
    coordinate_matrix_mask_lung.set_shape((4, 4))
    return (ct, pt, mask_lung1, mask_lung2, mask_gtvt, mask_gtvl,
            coordinate_matrix_ct, coordinate_matrix_pt,
            coordinate_matrix_mask_lung)


def parse_image(
    patient,
    path_nii,
    path_lung_mask_nii,
    ct_window_str="lung",
):
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
