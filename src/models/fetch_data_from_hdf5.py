from random import shuffle

import tensorflow as tf
import numpy as np
from numpy.random import randint

from src.models.data_augmentation import random_rotate


def get_bb_mask_voxel(mask):
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return np.array([x_min, y_min, z_min, x_max, y_max, z_max])


def unravel_mask(mask):
    return np.stack([mask == 1, mask == 2, mask == 3, mask == 4],
                    axis=-1).astype(np.float32)


def get_tf_data(
        file,
        clinical_df,
        output_shape=(256, 256),
        random_slice=True,
        centered_on_gtvt=False,
        random_shift=None,
        n_repeat=None,
        num_parallel_calls=None,
        oversample_plc_neg=False,
        patient_list=None,
        random_angle=None,
        output_type="segmentation",  # "segmentation+plc_status", "plc_status"
        return_complete_gtvl=False):
    """mask: mask_gtvt, mask_gtvl, mask_lung1, mask_lung2

    Args:
        file ([type]): [description]
        clinical_df ([type]): [description]
        output_shape (tuple, optional): [description]. Defaults to (256, 256).
        random_slice (bool, optional): [description]. Defaults to True.
        random_shift ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if patient_list is None:
        patient_list_copy = list(file.keys())
    else:
        patient_list_copy = patient_list.copy()
    if oversample_plc_neg:
        patient_list_plc_neg = [
            p for p in patient_list_copy
            if clinical_df.loc[int(p.split('_')[1]), "plc_status"] == 0
        ]
        patient_list_plc_pos = [
            p for p in patient_list_copy
            if clinical_df.loc[int(p.split('_')[1]), "plc_status"] == 1
        ]
        shuffle(patient_list_plc_neg)
        diff_patient = len(
            patient_list_plc_pos) - len(patient_list_plc_neg) * 2
        # patient_list = patient_list_plc_neg
        patient_list_copy.extend(patient_list_plc_neg)
        patient_list_copy.extend(patient_list_plc_neg[:diff_patient])
    shuffle(patient_list_copy)
    patient_ds = tf.data.Dataset.from_tensor_slices(patient_list_copy)
    if n_repeat:
        patient_ds = patient_ds.repeat(n_repeat)

    def f(patient):
        patient = patient.numpy().decode("utf-8")
        patient_id = patient.split('_')[1]
        plc_status = float(clinical_df.loc[int(patient_id), "plc_status"])
        sick_lung_axis = int(clinical_df.loc[int(patient_id),
                                             "sick_lung_axis"])
        image = h5_file[patient]["image"][()]
        mask = h5_file[patient]["mask"][()]
        # mask = unravel_mask(file[patient]["mask"][()])
        # w, h, n_slices, c = image.shape
        bb_lung = get_bb_mask_voxel(mask[..., 2] + mask[..., 3])
        center = ((bb_lung[:3] + bb_lung[3:]) // 2)[:2]
        bb_gtvl = get_bb_mask_voxel(mask[..., 1])
        bb_gtvt = get_bb_mask_voxel(mask[..., 0])
        if random_slice:
            # s = randint(bb_gtvt[2] + 1, bb_gtvt[5] - 1)
            if label_to_contain == "GTV T":
                s = randint(bb_gtvt[2] + 1, bb_gtvt[5] - 1)
            elif label_to_contain == "GTV L":
                s = randint(bb_gtvl[2] + 1, bb_gtvl[5] - 1)
            else:
                s = randint(1, n_slices - 1)
        else:
            if label_to_contain == "GTV T":
                s = (bb_gtvt[5] + bb_gtvt[2]) // 2
            elif label_to_contain == "GTV L":
                s = (bb_gtvl[5] + bb_gtvl[2]) // 2
            else:
                s = n_slices // 2
        if random_shift:
            center += np.array([
                randint(-random_shift, random_shift),
                randint(-random_shift, random_shift)
            ])

        r = [output_shape_image[i] // 2 for i in range(2)]
        mask = mask[center[0] - r[0]:center[0] + r[0],
                    center[1] - r[1]:center[1] + r[1], s, :]
        if plc_status == 1:
            gt_gtvl = mask[..., 1]
            mask_loss = (1 - mask[..., sick_lung_axis] + mask[..., 0] +
                         mask[..., 1])
        else:
            gt_gtvl = np.zeros(mask[..., 1].shape)
            mask_loss = np.ones_like(gt_gtvl)
        mask_loss[mask_loss > 0] = 1
        mask_loss[mask_loss <= 0] = 0
        if return_complete_gtvl:
            mask = np.stack(
                [
                    mask[..., 0], gt_gtvl, mask[..., 2] + mask[..., 3],
                    mask_loss, mask[..., 1]
                ],
                axis=-1,
            )
        else:
            mask = np.stack(
                [
                    mask[..., 0], gt_gtvl, mask[..., 2] + mask[..., 3],
                    mask_loss
                ],
                axis=-1,
            )
        image = np.squeeze(image[center[0] - r[0]:center[0] + r[0],
                                 center[1] - r[1]:center[1] + r[1], s, :])
        # image = standardize_image(image[center[0] - r[0]:center[0] + r[0],
        #                                 center[1] - r[1]:center[1] + r[1],
        #                                 s, :],
        #                           ct_clip_values,
        #                           pt_clip_values,
        #                           input_channels=output_shape_image[2])
        return image, mask, np.array([plc_status])

    def tf_f(patient):
        [image, mask,
         plc_status] = tf.py_function(f, [patient],
                                      [tf.float32, tf.float32, tf.float32])
        image.set_shape(output_shape_image)
        if return_complete_gtvl:
            mask.set_shape(output_shape_image[:2] + (5, ))
        else:
            mask.set_shape(output_shape_image[:2] + (4, ))
        plc_status.set_shape((1, ))
        return image, mask, plc_status

    if num_parallel_calls is not None:
        if num_parallel_calls == 'auto':
            num_parallel_calls = tf.data.experimental.AUTOTUNE

        out_ds = patient_ds.map(
            lambda patient: (*tf_f(patient), patient),
            num_parallel_calls=num_parallel_calls,
        )
        if random_angle:
            out_ds = out_ds.map(lambda x, y, p:
                                (*random_rotate(x, y, angle=random_angle), p),
                                num_parallel_calls=num_parallel_calls)
    else:
        out_ds = patient_ds.map(tf_f)
        if random_angle:
            out_ds = out_ds.map(lambda x, y, p:
                                (*random_rotate(x, y, angle=random_angle), p))

    if output_type == "segmentation":
        return out_ds.map(lambda x, y, p: (x, y))
    if output_type == "plc_status":
        return out_ds.map(lambda x, y, p: (x, p))
    if output_type == "segmentation+plc_status":
        return out_ds.map(lambda x, y, p: (x, (y, p)))
    else:
        raise ValueError(f"the output_type {output_type} is not defined")
