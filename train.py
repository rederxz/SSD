import os

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import SGD

from voc import load_voc_dataset, prepare
from modelV2 import SSD
from model_test import SSD_test
from loss import SSDLoss


# TODO: add lr schedule
# TODO: add mAP


def setup_tpu():
    tpu_address = os.environ.get("COLAB_TPU_ADDR")
    if tpu_address:
        tpu_address = "grpc://" + tpu_address
        tf.keras.backend.clear_session()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        return strategy


TEST = True

BATCH_SIZE = 32
EPOCH = 232  # iterations per epoch = 5011 / BATCH_SIZE

# data
ds_train, ds_test = load_voc_dataset()
ds_train = prepare(ds_train, batch_size=BATCH_SIZE, training=True)
ds_test = prepare(ds_test, batch_size=128)

boundaries = [154, 193]
values = [1e-3, 1e-4, 1e-5]
lr_schedule = PiecewiseConstantDecay(boundaries, values)
optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)

# metrics = [SSDLoss, mAP]
metrics = [SSDLoss, ]

if TEST:
    # with setup_tpu().scope():
    #     model = SSD_test()
    #     model.build((None, 300, 300, 3))
    #     model.compile(loss=SSDLoss, optimizer=optimizer, metrics=metrics)
    pass
else:
    with setup_tpu().scope():
        model = SSD()
        model.build((None, 300, 300, 3))
        model.compile(loss=SSDLoss, optimizer=optimizer, metrics=metrics)

# train
model.fit(ds_train, validation_data=ds_test, epochs=EPOCH)
