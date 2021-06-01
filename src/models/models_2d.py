import tensorflow as tf

from src.models.focal_loss import binary_focal_loss
from src.models.layers import ResidualLayer2D

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

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [
        tf.keras.layers.GlobalAveragePooling2D()(
            base_model.get_layer(name).output) for name in layer_names
    ]
    classification_model = tf.keras.Sequential([
        tf.keras.layers.Concatenate(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    output = classification_model(layers)

    # Create the feature extraction model
    return tf.keras.Model(inputs=base_model.input, outputs=output)


class Unet(tf.keras.Model):
    def __init__(self,
                 output_channels,
                 *args,
                 input_shape=(256, 256, 3),
                 **kwargs):
        super().__init__(*args, **kwargs)

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
        self.encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
        self.encoder.trainable = False

        self.up_blocks = [
            upsample(512, 3),  # 4x4 -> 8x8
            upsample(256, 3),  # 8x8 -> 16x16
            upsample(128, 3),  # 16x16 -> 32x32
            upsample(64, 3),  # 32x32 -> 64x64
            upsample(32, 3)
        ]

        self.last_segmentation = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="SAME"),
            tf.keras.layers.Conv2D(2, 1, activation="sigmoid"),
        ])

        self.last_plc = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="SAME"),
            tf.keras.layers.Conv2D(1, 1, activation="sigmoid"),
        ])

    def call(self, inputs, training=None):
        x1, x2, x3, x4, x5 = self.encoder(inputs, training=training)

        x6 = self.up_blocks[0](x5, training=training)
        x7 = self.up_blocks[1](tf.keras.layers.concatenate([x6, x4]),
                               training=training)
        x8 = self.up_blocks[2](tf.keras.layers.concatenate([x7, x3]),
                               training=training)
        x9 = self.up_blocks[3](tf.keras.layers.concatenate([x8, x2]),
                               training=training)
        x10 = self.up_blocks[4](tf.keras.layers.concatenate([x9, x1]),
                                training=training)
        x_seg = self.last_segmentation(x10)
        x_plc = self.last_plc(x10)

        return tf.stack([x_seg[..., 0], x_plc[..., 0], x_seg[..., 1]], axis=-1)


class UnetClassif(Unet):
    def __init__(self, output_channels, *args, input_shape, **kwargs):
        super().__init__(output_channels,
                         *args,
                         input_shape=input_shape,
                         **kwargs)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        self.gaps = [
            tf.keras.layers.GlobalAveragePooling2D() for _ in range(5)
        ]

    def call(self, inputs, training=None):
        x1, x2, x3, x4, x5 = self.encoder(inputs, training=training)

        x6 = self.up_blocks[0](x5, training=training)
        x7 = self.up_blocks[1](tf.keras.layers.concatenate([x6, x4]),
                               training=training)
        x8 = self.up_blocks[2](tf.keras.layers.concatenate([x7, x3]),
                               training=training)
        x9 = self.up_blocks[3](tf.keras.layers.concatenate([x8, x2]),
                               training=training)
        x10 = self.up_blocks[4](tf.keras.layers.concatenate([x9, x1]),
                                training=training)

        x1, x2, x3, x4, x5 = [
            self.gaps[i](x) for i, x in enumerate([x1, x2, x3, x4, x5])
        ]

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


class UnetIantsen(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
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
        self.last_gtvt = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='SAME'),
        ])
        self.last_gtvl = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='SAME'),
        ])
        self.last_lung = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='SAME'),
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
            x = block(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip))
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        x_gtvt = self.last_gtvt(x)
        x_gtvl = self.last_gtvl(x)
        x_lung = self.last_lung(x)
        return tf.keras.layers.concatenate([x_gtvt, x_gtvl, x_lung])
