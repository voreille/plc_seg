import tensorflow as tf
import numpy as np
import pandas as pd
from numpy.random import randint


def get_tf_data(
    file,
    clinical_df,
    output_shape=(256, 256),
    random_slice=True,
    random_shift=None,
    n_repeat=None,
    num_parallel_calls=None,
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
    patient_list = list(file.keys())
    patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)
    if n_repeat:
        patient_ds = patient_ds.repeat(n_repeat)

    def f(patient):
        patient = patient.numpy().decode("utf-8")
        patient_id = patient.split('_')[1]
        plc_status = float(clinical_df.loc[int(patient_id), "plc_status"])
        image = file[patient]["image"][()]
        mask = file[patient]["mask"][()]
        w, h, n_slices, c = image.shape
        if random_slice:
            s = randint(n_slices)
        else:
            s = n_slices // 2
        if random_shift:
            center = (w // 2 + randint(-random_shift, random_shift),
                      h // 2 + randint(-random_shift, random_shift))
        else:
            center = (w // 2, h // 2)

        r = [output_shape[i] // 2 for i in range(2)]
        return image[center[0] - r[0]:center[0] + r[0],
                     center[1] - r[1]:center[1] + r[1],
                     s, :], mask[center[0] - r[0]:center[0] + r[0],
                                 center[1] - r[1]:center[1] + r[1],
                                 s, :], np.array([plc_status])

    def tf_f(patient):
        [image, mask,
         plc_status] = tf.py_function(f, [patient],
                                      [tf.float32, tf.float32, tf.float32])
        image.set_shape(output_shape + (3, ))
        mask.set_shape(output_shape + (4, ))
        plc_status.set_shape((1, ))
        return image, mask, plc_status

    if num_parallel_calls is not None:
        if num_parallel_calls == 'auto':
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        return patient_ds.map(
            tf_f,
            num_parallel_calls=num_parallel_calls,
        )
    else:
        return patient_ds.map(tf_f)
