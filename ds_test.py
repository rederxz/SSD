import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Layer
from tensorflow.keras.applications import VGG16


from modelV2 import SSD
from voc import load_voc_dataset, prepare
from loss import SSDLoss

ds_train, ds_test = load_voc_dataset()

ds_train = prepare(ds_train, training=True)

print(ds_train.take(2))

# for i in ds_train.take(2):
#     print(i)
#     break


#
#
# dummy = tf.ones((1, 300, 300, 3))
#
# model = SSD()
#
# _ = model(dummy)
#
# print(_)