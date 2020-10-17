import numpy as np
import tensorflow as tf
from tensorflow import keras

class OrdinalOutput(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.t0 = tf.constant(-np.inf, shape=(1, 1))
        self.tK = tf.constant(np.inf, shape=(1, 1))
        super(OrdinalOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] == 1 and len(input_shape) <= 2
        # TODO: handle input with extra dimensions and
        # different regression axis
        # e.g. sequence of time-major or not
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(1, self.output_dim - 1),
            initializer=self.sorted_initializer('glorot_uniform'),
            trainable=True)
        # 1. Overwritting `self.thresholds` attribute causes
        # TF not to maintain the added weight.
        # 2. Calling tf.concat here is eager by default
        # and causes the result to be treated as constant.
        # (found by looking at Tensor Board graph)
        # Moved to `call`.
        super(OrdinalOutput, self).build(input_shape)

    def call(self, x):
        upper = tf.concat([self.thresholds, self.tK],  axis=-1)
        lower = tf.concat([self.t0, self.thresholds],  axis=-1)
        output = tf.sigmoid(upper - x) - tf.sigmoid(lower - x)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def sorted_initializer(self, initializer):
        # Returns a function that returns a sorted
        # initialization based on an initializer string
        initializer = keras.initializers.get(initializer)

        def sorter(shape, dtype=None):
            # Returns a sorted initialization
            return tf.sort(initializer(shape, dtype))
        return sorter
