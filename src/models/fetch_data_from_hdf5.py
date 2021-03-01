from random import shuffle

import tensorflow as tf
import numpy as np
from numpy.random import randint


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
):
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
        patient_list = list(file.keys())
    if oversample_plc_neg:
        patient_list_plc_neg = [
            p for p in patient_list
            if clinical_df.loc[int(p.split('_')[1]), "plc_status"] == 0
        ]
        patient_list_plc_pos = [
            p for p in patient_list
            if clinical_df.loc[int(p.split('_')[1]), "plc_status"] == 1
        ]
        shuffle(patient_list_plc_neg)
        diff_patient = len(patient_list_plc_pos) - len(patient_list_plc_neg) * 2
        # patient_list = patient_list_plc_neg
        patient_list.extend(patient_list_plc_neg)
        patient_list.extend(patient_list_plc_neg[:diff_patient])
    shuffle(patient_list)
    patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)
    if n_repeat:
        patient_ds = patient_ds.repeat(n_repeat)

    def f(patient):
        patient = patient.numpy().decode("utf-8")
        patient_id = patient.split('_')[1]
        plc_status = float(clinical_df.loc[int(patient_id), "plc_status"])
        sick_lung_axis = int(clinical_df.loc[int(patient_id),
                                             "sick_lung_axis"])
        image = file[patient]["image"][()]
        mask = file[patient]["mask"][()]
        # mask = unravel_mask(file[patient]["mask"][()])
        w, h, n_slices, c = image.shape
        bb_lung = get_bb_mask_voxel(mask[..., 2] + mask[..., 3])
        center = ((bb_lung[:3] + bb_lung[3:]) // 2)[:2]
        bb_gtvl = get_bb_mask_voxel(mask[..., 1])
        bb_gtvt = get_bb_mask_voxel(mask[..., 0])
        if random_slice:
            s = randint(bb_gtvl[2], bb_gtvl[5])
        elif random_slice and plc_status == 0:
            s = randint(bb_gtvt[2], bb_gtvt[5])
        elif centered_on_gtvt:
            s = (bb_gtvt[5] + bb_gtvt[2]) // 2
        else:
            s = (bb_gtvl[5] + bb_gtvl[2]) // 2
        if random_shift:
            center += np.array([
                randint(-random_shift, random_shift),
                randint(-random_shift, random_shift)
            ])

        r = [output_shape[i] // 2 for i in range(2)]
        return image[center[0] - r[0]:center[0] + r[0],
                     center[1] - r[1]:center[1] + r[1],
                     s, :], mask[center[0] - r[0]:center[0] + r[0],
                                 center[1] - r[1]:center[1] + r[1],
                                 s, :], np.array([plc_status]), np.array(
                                     [sick_lung_axis], dtype=np.int8)

    def tf_f(patient):
        [image, mask, plc_status, sick_lung_axis
         ] = tf.py_function(f, [patient],
                            [tf.float32, tf.float32, tf.float32, tf.int8])
        image.set_shape(output_shape + (3, ))
        mask.set_shape(output_shape + (4, ))
        plc_status.set_shape((1, ))
        sick_lung_axis.set_shape((1, ))
        return image, mask, plc_status, sick_lung_axis

    if num_parallel_calls is not None:
        if num_parallel_calls == 'auto':
            num_parallel_calls = tf.data.experimental.AUTOTUNE

        return patient_ds.map(
            tf_f,
            num_parallel_calls=num_parallel_calls,
        )
    else:
        return patient_ds.map(tf_f)
