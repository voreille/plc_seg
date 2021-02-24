import tensorflow as tf
import numpy as np
from numpy.random import randint


def get_tf_data(file,
                output_shape=(256, 256),
                random_slice=True,
                random_shift=None):
    patient_list = list(file.keys())
    patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)

    def f(patient):
        patient = patient.numpy().decode("utf-8")
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

        # print(
        #     f"For {patient} the shapes are: image {image.shape}, mask {mask.shape} and s {s}"
        # )
        r = [output_shape[i] // 2 for i in range(2)]
        return image[center[0] - r[0]:center[0] + r[0],
                     center[1] - r[1]:center[1] + r[1],
                     s, :], mask[center[0] - r[0]:center[0] + r[0],
                                 center[1] - r[1]:center[1] + r[1], s, :]

    def tf_f(patient):
        [image, mask] = tf.py_function(f, [patient], [tf.float32, tf.float32])
        image.set_shape(output_shape + (3, ))
        mask.set_shape(output_shape + (4, ))
        return image, mask

    return patient_ds.map(tf_f)
