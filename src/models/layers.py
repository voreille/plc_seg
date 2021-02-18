import tensorflow as tf


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, *args, activation='relu', **kwargs):
        super().__init__()
        self.filters = args[0]
        self.conv = tf.keras.layers.Conv3D(*args,
                                           **kwargs,
                                           activation=activation)
        self.activation = activation
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = None
        self.proj = None

    def build(self, input_shape):
        self.c_in = input_shape[1]
        if input_shape[1] != self.filters:
            self.proj = tf.keras.layers.Conv3D(self.filters,
                                               1,
                                               activation=self.activation)
            self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        if self.proj:
            return self.bn_1(self.conv(x)) + self.bn_2(self.proj(x))
        else:
            return self.bn_1(self.conv(x)) + x


class SENormLayer(tf.keras.layers.Layer):
    pass
