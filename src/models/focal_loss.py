from functools import partial

import tensorflow as tf

from src.models.utils_focal_loss import check_bool, check_float

_EPSILON = tf.keras.backend.epsilon()


def binary_focal_loss(y_true,
                      y_pred,
                      gamma=2,
                      *,
                      pos_weight=None,
                      from_logits=False,
                      label_smoothing=None):

    gamma = check_float(gamma, name='gamma', minimum=0)
    pos_weight = check_float(pos_weight,
                             name='pos_weight',
                             minimum=0,
                             allow_none=True)
    from_logits = check_bool(from_logits, name='from_logits')
    label_smoothing = check_float(label_smoothing,
                                  name='label_smoothing',
                                  minimum=0,
                                  maximum=1,
                                  allow_none=True)
    # Ensure predictions are a floating point tensor; converting labels to a
    # tensor will be done in the helper functions
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.float32)

    # Delegate per-example loss computation to helpers depending on whether
    # predictions are logits or probabilities
    if from_logits:
        return _binary_focal_loss_from_logits(labels=y_true,
                                              logits=y_pred,
                                              gamma=gamma,
                                              pos_weight=pos_weight,
                                              label_smoothing=label_smoothing)
    else:
        return _binary_focal_loss_from_probs(labels=y_true,
                                             p=y_pred,
                                             gamma=gamma,
                                             pos_weight=pos_weight,
                                             label_smoothing=label_smoothing)


def _binary_focal_loss_from_probs(labels, p, gamma, pos_weight,
                                  label_smoothing):
    """Compute focal loss from probabilities.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's: binary class labels.
    p : tf.Tensor
        Estimated probabilities for the positive class.
    gamma : float
        Focusing parameter.
    pos_weight : float or None
        If not None, losses for the positive class will be scaled by this
        weight.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    Returns
    -------
    tf.Tensor
        The loss for each example.
    """
    # Predicted probabilities for the negative class
    q = 1 - p

    # For numerical stability (so we don't inadvertently take the log of 0)
    p = tf.math.maximum(p, _EPSILON)
    q = tf.math.maximum(q, _EPSILON)

    # Loss for the positive examples
    pos_loss = -(q**gamma) * tf.math.log(p)
    if pos_weight is not None:
        pos_loss *= pos_weight

    # Loss for the negative examples
    neg_loss = -(p**gamma) * tf.math.log(q)

    # Combine loss terms
    if label_smoothing is None:
        # labels = tf.dtypes.cast(labels, dtype=tf.bool)
        loss = tf.where(labels > 0.5, x=pos_loss, y=neg_loss)
    else:
        labels = _process_labels(labels=labels,
                                 label_smoothing=label_smoothing,
                                 dtype=p.dtype)
        loss = labels * pos_loss + (1 - labels) * neg_loss

    return loss


def _binary_focal_loss_from_logits(labels, logits, gamma, pos_weight,
                                   label_smoothing):
    """Compute focal loss from logits using a numerically stable formula.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's: binary class labels.
    logits : tf.Tensor
        Logits for the positive class.
    gamma : float
        Focusing parameter.
    pos_weight : float or None
        If not None, losses for the positive class will be scaled by this
        weight.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    Returns
    -------
    tf.Tensor
        The loss for each example.
    """
    labels = _process_labels(labels=labels,
                             label_smoothing=label_smoothing,
                             dtype=logits.dtype)

    # Compute probabilities for the positive class
    p = tf.math.sigmoid(logits)

    # Without label smoothing we can use TensorFlow's built-in per-example cross
    # entropy loss functions and multiply the result by the modulating factor.
    # Otherwise, we compute the focal loss ourselves using a numerically stable
    # formula below
    if label_smoothing is None:
        # The labels and logits tensors' shapes need to be the same for the
        # built-in cross-entropy functions. Since we want to allow broadcasting,
        # we do some checks on the shapes and possibly broadcast explicitly
        # Note: tensor.shape returns a tf.TensorShape, whereas tf.shape(tensor)
        # returns an int tf.Tensor; this is why both are used below
        labels_shape = labels.shape
        logits_shape = logits.shape
        if not labels_shape.is_fully_defined() or labels_shape != logits_shape:
            labels_shape = tf.shape(labels)
            logits_shape = tf.shape(logits)
            shape = tf.broadcast_dynamic_shape(labels_shape, logits_shape)
            labels = tf.broadcast_to(labels, shape)
            logits = tf.broadcast_to(logits, shape)
        if pos_weight is None:
            loss_func = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            loss_func = partial(tf.nn.weighted_cross_entropy_with_logits,
                                pos_weight=pos_weight)
        loss = loss_func(labels=labels, logits=logits)
        modulation_pos = (1 - p)**gamma
        modulation_neg = p**gamma
        mask = tf.dtypes.cast(labels, dtype=tf.bool)
        modulation = tf.where(mask, modulation_pos, modulation_neg)
        return modulation * loss

    # Terms for the positive and negative class components of the loss
    pos_term = labels * ((1 - p)**gamma)
    neg_term = (1 - labels) * (p**gamma)

    # Term involving the log and ReLU
    log_weight = pos_term
    if pos_weight is not None:
        log_weight *= pos_weight
    log_weight += neg_term
    log_term = tf.math.log1p(tf.math.exp(-tf.math.abs(logits)))
    log_term += tf.nn.relu(-logits)
    log_term *= log_weight

    # Combine all the terms into the loss
    loss = neg_term * logits + log_term
    return loss


def _process_labels(labels, label_smoothing, dtype):
    """Pre-process a binary label tensor, maybe applying smoothing.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    dtype : tf.dtypes.DType
        Desired type of the elements of `labels`.
    Returns
    -------
    tf.Tensor
        The processed labels.
    """
    labels = tf.dtypes.cast(labels, dtype=dtype)
    if label_smoothing is not None:
        labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
    return labels
