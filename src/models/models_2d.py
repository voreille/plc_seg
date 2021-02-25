import tensorflow as tf

from src.models.losses_2d import dice_coe_1

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
                                           padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def get_mask(y_true):
    if tf.reduce_sum(y_true[..., 2] * y_true[..., 1]) != 0:
        sick_lung = y_true[..., 2]
    else:
        sick_lung = y_true[..., 3]
    mask = 1 - (sick_lung - y_true[..., 0] - y_true[..., 2])
    return mask


def custom_loss(y_true, y_pred, plc_status):
    """ 0: gtvt, 1: gtvl, 2: lung1, 3: lung2

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
    """
    mask = tf.map_fn(fn=get_mask, elems=y_true)
    mask = tf.where(mask > 0.9, x=1.0, y=0.0)
    # print(f"hey its the shape of the mask {mask.shape}")

    lung = y_true[..., 2] + y_true[..., 3]

    loss = -(dice_coe_1(y_true[..., 0], y_pred[..., 0]) +
             dice_coe_1(lung, y_pred[..., 2]))

    return 3 + tf.reduce_mean(
        tf.where(
            plc_status >= 0.1,
            x=loss - dice_coe_1(mask * y_true[..., 1], mask * y_pred[..., 1]),
            y=loss -
            dice_coe_1(tf.zeros(tf.shape(y_pred[..., 1])), y_pred[..., 1]),
        ))


loss_tracker = tf.keras.metrics.Mean(name="loss")
mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y, plc_status = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = custom_loss(y, y_pred, plc_status)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # # Compute our own metrics
        # mae_metric.update_state(y, y_pred)
        # return {"loss": loss_tracker.result(), "mae": mae_metric.result()}
        return {"loss": loss}

    def test_step(self, data):
        # Unpack the data
        x, y, plc_status = data
        # Compute predictions
        y_pred = self(x, training=False)

        return {"loss_val": custom_loss(y, y_pred, plc_status)}

    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [loss_tracker, mae_metric]
