import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Layer
from tensorflow.keras.applications import VGG16


# TODO: Add weight decay
# TODO: Add layer-wise learning rates


class SSD(Model):
    def __init__(self):
        super(Model, self).__init__()

    def call(self, x, **kwargs):
        batch_size = tf.shape(x)[0]
        return {
            'offsets': tf.ones((batch_size, 8732, 4)),
            'classes': tf.ones((batch_size, 8732, 21))
        }