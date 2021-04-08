import tensorflow as tf

from src.models.losses_2d import dice_coe_1, dice_coe_1_hard
from src.models.focal_loss import binary_focal_loss

import matplotlib.pyplot as plt

OUTPUT_CHANNELS = 3


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters,
                                        size,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_model(output_channels, input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels,
                                           3,
                                           strides=2,
                                           activation="sigmoid",
                                           padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def gtvl_loss(
    y_true,
    y_pred,
    plc_status,
    sick_lung_axis,
    pos_weight=1.0,
    scaling=10.0,
):
    mask = 1 - tf.where(sick_lung_axis[..., tf.newaxis] == 3,
                        x=y_true[..., 3],
                        y=y_true[..., 2]) + y_true[..., 1] + y_true[..., 0]
    mask = tf.where(mask > 0.5, x=1.0, y=0.0)
    n_elems = tf.reduce_sum(mask, axis=(1, 2))
    return scaling * tf.where(
        tf.squeeze(plc_status) >= 0.5,
        x=tf.reduce_sum(binary_focal_loss(
            y_true[..., 1], y_pred[..., 1], pos_weight=pos_weight) * mask,
                        axis=(1, 2)) / n_elems,
        y=tf.reduce_mean(binary_focal_loss(tf.zeros(tf.shape(y_pred[..., 1])),
                                           y_pred[..., 1],
                                           pos_weight=pos_weight),
                         axis=(1, 2)))


def classif_model(n_class, input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False)


    base_model.trainable = False


    return tf.keras.Model(inputs=inputs, outputs=x)
