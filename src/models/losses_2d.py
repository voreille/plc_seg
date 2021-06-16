import tensorflow as tf
import tensorflow.keras.backend as K

from src.models.focal_loss import binary_focal_loss


def gtvl_loss(y_true, y_pred, pos_weight=1.0):
    n_elems = tf.reduce_sum(y_true[..., 3], axis=(1, 2))
    return tf.reduce_sum(
        binary_focal_loss(
            y_true[..., 1], y_pred[..., 1], pos_weight=pos_weight) *
        y_true[..., 3],
        axis=(1, 2),
    ) / n_elems


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self,
                 pos_weight=1.0,
                 w_lung=1,
                 w_gtvt=1,
                 w_gtvl=4,
                 name="custom_loss"):
        super().__init__(name=name)
        self.pos_weight = pos_weight
        self.w_lung = w_lung
        self.w_gtvt = w_gtvt
        self.w_gtvl = w_gtvl

    def _gtvl_loss(self, y_true, y_pred):
        n_elems = tf.reduce_sum(y_true[..., 3], axis=(1, 2))
        return tf.reduce_sum(
            binary_focal_loss(
                y_true[..., 1], y_pred[..., 1], pos_weight=self.pos_weight) *
            y_true[..., 3],
            axis=(1, 2),
        ) / n_elems

    def call(self, y_true, y_pred):
        return (self.w_gtvt *
                (1 - dice_coe_1(y_true[..., 0], y_pred[..., 0])) +
                self.w_lung *
                (1 - dice_coe_1(y_true[..., 2], y_pred[..., 2])) +
                self.w_gtvl * self._gtvl_loss(y_true, y_pred))


def masked_focal_loss(y_true, y_pred, mask, gamma=2):
    n_pos = tf.reduce_sum(y_true)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred) *
                          (1 - y_pred)**gamma) / n_pos


def dice_coe_1_hard(y_true, y_pred, loss_type='sorensen', smooth=1.):
    return dice_coe_1(y_true,
                      tf.cast(y_pred > 0.5, tf.float32),
                      loss_type=loss_type,
                      smooth=smooth)


def dice_coe_1(y_true, y_pred, loss_type='jaccard', smooth=1., axis=(1, 2)):
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    if loss_type == 'jaccard':
        union = tf.reduce_sum(
            tf.square(y_pred),
            axis=axis,
        ) + tf.reduce_sum(tf.square(y_true), axis=axis)

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred, axis=axis) + tf.reduce_sum(y_true,
                                                                 axis=axis)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)
    return (2. * intersection + smooth) / (union + smooth)


def dice_coe_loss(y_true, y_pred, loss_type='jaccard', smooth=1.):
    return 1 - dice_coe(y_true, y_pred, loss_type=loss_type, smooth=smooth)


def dice_coe_hard(y_true, y_pred, loss_type='sorensen', smooth=1.):
    return dice_coe(y_true,
                    tf.cast(y_pred > 0.5, tf.float32),
                    loss_type=loss_type,
                    smooth=smooth)


def dice_coe(y_true, y_pred, loss_type='jaccard', smooth=1., axis=(1, 2)):
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    n_classes = y_pred.shape[-1]
    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred), axis=axis) + tf.reduce_sum(
            tf.square(y_true), axis=axis)

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred, axis=axis) + tf.reduce_sum(y_true,
                                                                 axis=axis)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)
    return tf.reduce_mean(
        tf.reduce_sum((2. * intersection + smooth) /
                      (union + smooth), axis=-1)) / n_classes


def focal_loss(y_true, y_pred, gamma=2):
    n_pos = tf.reduce_sum(y_true)
    bs = y_true.shape[0]
    return -tf.reduce_sum(y_true * tf.math.log(y_pred) *
                          (1 - y_pred)**gamma) / bs / n_pos


def focal_loss_fix(y_true, y_pred, gamma=2):
    n_pos = tf.reduce_sum(y_true)
    bs = y_true.shape[0]
    return -tf.reduce_sum(0.25 * y_true * tf.math.log(y_pred) *
                          (1 - y_pred)**gamma) / bs / n_pos


def binary_focal_loss_custom(y_true, y_pred, gamma=2):
    alpha = 1 / K.sum(y_true, axis=(1, 2, 3, 4))
    beta = 1 / K.sum(tf.cast(tf.equal(y_true, 0.0), dtype=tf.float32),
                     axis=(1, 2, 3, 4))
    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    loss = (y_true * (-alpha) * (1 - y_pred)**gamma * K.log(y_pred)) - (
        (1 - y_true) * (beta) * y_pred**gamma * K.log(1 - y_pred))
    return K.mean(K.sum(loss, axis=(1, 2, 3, 4))) / 2


def binary_focal_loss_fixed(y_true, y_pred, alpha=1, gamma=2):
    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1 - p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.mean(loss)
    return loss
