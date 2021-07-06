from scipy.sparse import base
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.python.keras.backend import dropout
from tensorflow_addons.losses import sigmoid_focal_crossentropy

from src.models.losses_2d import dice_coe_1, dice_coe_1_hard
from src.models.layers import ResidualLayer2D

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


def unetclassif_model(output_channels, input_shape=(256, 256, 3)):
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
    x_classif = base_model(x)

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Classifier
    classifier = tf.keras.Sequential(name="classifier")
    classifier.add(tf.keras.layers.GlobalAveragePooling2D())
    classifier.add(tf.keras.layers.Dense(2, activation="softmax"))
    x_classif = classifier(x_classif)

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

    return tf.keras.Model(inputs=inputs, outputs=[x, x_classif])


def classif_model(output_channels,
                  input_shape=(256, 256, 3),
                  dropout_rate=0.2):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False)

    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    x = base_model(x)

    # Classifier
    classifier = tf.keras.Sequential(name="classifier")
    classifier.add(tf.keras.layers.GlobalAveragePooling2D())
    classifier.add(tf.keras.layers.Dropout(dropout_rate))
    classifier.add(tf.keras.layers.Dense(2, activation="softmax"))

    return tf.keras.Model(inputs=inputs, outputs=classifier(x))

        return x10, self.classifier(
            tf.keras.layers.concatenate([x1, x2, x3, x4, x5]))

class UpBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 *args,
                 upsampling_factor=1,
                 filters_output=24,
                 n_conv=2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.upsampling_factor = upsampling_factor
        self.conv = tf.keras.Sequential()
        for k in range(n_conv):
            self.conv.add(
                tf.keras.layers.Conv2D(filters,
                                       3,
                                       padding='SAME',
                                       activation='relu'), )
        self.trans_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                          3,
                                                          strides=(2, 2),
                                                          padding='SAME',
                                                          activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        if upsampling_factor != 1:
            self.upsampling = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters_output,
                                       1,
                                       padding='SAME',
                                       activation='relu'),
                tf.keras.layers.UpSampling2D(size=(upsampling_factor,
                                                   upsampling_factor)),
            ])
        else:
            self.upsampling = None

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.trans_conv(x)
        x = self.concat([x, skip])
        x = self.conv(x)
        if self.upsampling:
            return x, self.upsampling(x)
        else:
            return x


class Unet(tf.keras.Model):
    def __init__(self, *args, output_channels=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_stack = [
            self.get_first_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
            self.get_down_block(384),
        ]

        self.up_stack = [
            UpBlock(192, upsampling_factor=8),
            UpBlock(96, upsampling_factor=4),
            UpBlock(48, upsampling_factor=2),
            UpBlock(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation='sigmoid',
                                   padding='SAME'),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 7, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last(x)


class UnetClassif(tf.keras.Model):
    def __init__(self, *args, output_channels=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_stack = [
            self.get_first_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
            self.get_down_block(384),
        ]

        self.up_stack = [
            UpBlock(192, upsampling_factor=8),
            UpBlock(96, upsampling_factor=4),
            UpBlock(48, upsampling_factor=2),
            UpBlock(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation='sigmoid',
                                   padding='SAME'),
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 7, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x, training=training)
            skips.append(x)

        x_classif = self.classifier(x, training=training)
        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last(x), x_classif
