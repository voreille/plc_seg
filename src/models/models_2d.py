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


def get_mask(y_true, sick_lung_axis):
    if sick_lung_axis == 9:
        return tf.ones(y_true[..., 3].shape)
    else:
        return 1 - (y_true[..., sick_lung_axis] - y_true[..., 1] -
                    y_true[..., 0])


def custom_loss(
    y_true,
    y_pred,
    plc_status,
    sick_lung_axis,
    pos_weight=1.0,
    w_lung=1,
    w_gtvt=1,
    w_gtvl=4,
):
    """ 0: gtvt, 1: gtvl, 2: lung1, 3: lung2

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
    """
    mask = 1 - tf.where(sick_lung_axis[..., tf.newaxis] == 3,
                        x=y_true[..., 3],
                        y=y_true[..., 2]) + y_true[..., 1] + y_true[..., 0]
    mask = tf.where(mask > 0.5, x=1.0, y=0.0)
    n_elems = tf.reduce_sum(mask, axis=(1, 2))
    # print(f"hey its the shape of the mask {mask.shape}")

    loss = (w_gtvt * (1 - dice_coe_1(y_true[..., 0], y_pred[..., 0])) +
            w_lung *
            (1 - dice_coe_1(y_true[..., 2] + y_true[..., 3], y_pred[..., 2])))

    return 10 * tf.reduce_mean(
        tf.where(
            tf.squeeze(plc_status) >= 0.5,
            x=loss + w_gtvl * tf.reduce_sum(binary_focal_loss(
                y_true[..., 1], y_pred[..., 1], pos_weight=pos_weight) * mask,
                                            axis=(1, 2)) / n_elems,
            y=loss + w_gtvl * tf.reduce_mean(
                binary_focal_loss(tf.zeros(tf.shape(y_pred[..., 1])),
                                  y_pred[..., 1],
                                  pos_weight=pos_weight),
                axis=(1, 2)))) / (w_lung + w_gtvl + w_gtvt)


loss_tracker = tf.keras.metrics.Mean(name="loss")
gtvt_dice_tracker = tf.keras.metrics.Mean(name="gtvt_dice")
lung_dice_tracker = tf.keras.metrics.Mean(name="lung_dice")

val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
val_gtvt_dice_tracker = tf.keras.metrics.Mean(name="val_gtvt_dice")
val_lung_dice_tracker = tf.keras.metrics.Mean(name="val_lung_dice")


class CustomModel(tf.keras.Model):
    def __init__(
        self,
        *args,
        alpha=1.0,
        w_lung=1.0,
        w_gtvt=1.0,
        w_gtvl=4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.w_lung = w_lung
        self.w_gtvt = w_gtvt
        self.w_gtvl = w_gtvl

    def train_step(self, data):
        x, y, plc_status, sick_lung_axis = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = custom_loss(
                y,
                y_pred,
                plc_status,
                sick_lung_axis,
                pos_weight=self.alpha,
                w_lung=self.w_lung,
                w_gtvt=self.w_gtvt,
                w_gtvl=self.w_gtvl,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        gtvt_dice_tracker.update_state(
            tf.reduce_mean(dice_coe_1_hard(y[..., 0], y_pred[..., 0])))
        lung_dice_tracker.update_state(
            tf.reduce_mean(
                dice_coe_1_hard(y[..., 2] + y[..., 3], y_pred[..., 2])))
        return {
            "loss": loss_tracker.result(),
            "dice_gtvt": gtvt_dice_tracker.result(),
            "dice_lung": lung_dice_tracker.result()
        }

    def test_step(self, data):
        # Unpack the data
        x, y, plc_status, sick_lung_axis = data
        # Compute predictions
        y_pred = self(x, training=False)

        val_loss_tracker.update_state(
            custom_loss(y, y_pred, plc_status, sick_lung_axis))
        val_gtvt_dice_tracker.update_state(
            tf.reduce_mean(dice_coe_1_hard(y[..., 0], y_pred[..., 0])))
        val_lung_dice_tracker.update_state(
            tf.reduce_mean(
                dice_coe_1_hard(y[..., 2] + y[..., 3], y_pred[..., 2])))
        return {
            "loss_val": val_loss_tracker.result(),
            "dice_gtvt": val_gtvt_dice_tracker.result(),
            "dice_lung": val_lung_dice_tracker.result()
        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            loss_tracker, gtvt_dice_tracker, lung_dice_tracker,
            val_loss_tracker, val_gtvt_dice_tracker, val_lung_dice_tracker
        ]
