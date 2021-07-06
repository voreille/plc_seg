import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike


def gtvl_loss(y_true, y_pred, scaling=1.0, alpha=1.0):
    n_elems = tf.reduce_sum(y_true[..., 3], axis=(1, 2))
    return scaling * tf.reduce_sum(
        sigmoid_focal_crossentropy(y_true[..., 1], y_pred[..., 1], alpha=alpha)
        * y_true[..., 3],
        axis=(1, 2),
    ) / n_elems


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self,
                 alpha=1.0,
                 w_lung=1,
                 w_gtvt=1,
                 w_gtvl=4,
                 s_gtvl=10,
                 loss_type="masked",
                 name="custom_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.w_lung = w_lung
        self.w_gtvt = w_gtvt
        self.w_gtvl = w_gtvl
        self.s_gtvl = s_gtvl
        self.loss_type = loss_type

    def _gtvl_loss(self, y_true, y_pred):
        if self.loss_type == "masked":
            return self._gtvl_loss_masked(y_true, y_pred)
        elif self.loss_type == "pseudolabel":
            return self._gtvl_loss_pseudolabel(y_true, y_pred)

    def _gtvl_loss_masked(self, y_true, y_pred):
        n_elems = tf.reduce_sum(y_true[..., 3], axis=(1, 2))
        return self.s_gtvl * tf.reduce_sum(
            sigmoid_focal_crossentropy(
                y_true[..., 1],
                y_pred[..., 1],
                alpha=self.alpha,
            ) * y_true[..., 3],
            axis=(1, 2),
        ) / n_elems

    def _gtvl_loss_pseudolabel(self, y_true, y_pred):
        return self.s_gtvl * tf.reduce_mean(
            tf.where(y_true[..., 3] == 1,
                     x=sigmoid_focal_crossentropy(
                         y_true[..., 1],
                         y_pred[..., 1],
                         alpha=self.alpha,
                     ),
                     y=sigmoid_focal_crossentropy(
                         y_pred[..., 1],
                         y_pred[..., 1],
                         alpha=self.alpha,
                     )),
            axis=(1, 2),
        )

    def call(self, y_true, y_pred):
        l1 = self._gtvl_loss(y_true, y_pred)
        return (self.w_gtvt *
                (1 - dice_coe_1(y_true[..., 0], y_pred[..., 0])) +
                self.w_lung *
                (1 - dice_coe_1(y_true[..., 2], y_pred[..., 2])) +
                self.w_gtvl * l1) / (self.w_gtvl + self.w_gtvt + self.w_lung)


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


@tf.function
def sigmoid_focal_crossentropy(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.25,
    gamma: FloatTensorLike = 2.0,
    from_logits: bool = False,
) -> tf.Tensor:
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.

    Copied from https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/losses/focal_loss.py#L84-L142
    and modified to avoid reduction
    """
    if gamma and gamma < 0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return, modified by Val
    return alpha_factor * modulating_factor * ce