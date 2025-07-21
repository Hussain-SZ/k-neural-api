# In file: k/cai/color_layers.py
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Cai")
class RgbToLab(layers.Layer):
    def __init__(self, **kwargs):
        super(RgbToLab, self).__init__(**kwargs)

    def call(self, inputs):
        # The input tensor is float32, which is what rgb_to_lab expects.
        lab_image = tfio.experimental.color.rgb_to_lab(inputs)
        return lab_image

    def get_config(self):
        return super(RgbToLab, self).get_config()
