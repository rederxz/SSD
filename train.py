from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

from voc import load_voc_dataset, prepare
from modelV2 import SSD
from loss import SSDLoss


# TODO: add lr schedule
# TODO: add mAP


def lr_scheduler(epoch, lr):
    if epoch < 154:
        return 1e-3
    elif epoch < 193:
        return 1e-4
    else:
        return 1e-5


BATCH_SIZE = 32
EPOCH = 232  # iterations per epoch = 5011 / BATCH_SIZE

# data
ds_train, ds_test = load_voc_dataset()
ds_train = prepare(ds_train, batch_size=BATCH_SIZE, training=True)
ds_test = prepare(ds_test, batch_size=128)

# model, loss, optimizer, metric
model = SSD()
model.build((None, 300, 300, 3))

lr_callback = LearningRateScheduler(lr_scheduler)
optimizer = SGD(learning_rate=1e-3, momentum=0.9)

# metrics = [SSDLoss, mAP]
metrics = [SSDLoss, ]

model.compile(loss=SSDLoss, optimizer=optimizer, metrics=metrics)

# train
model.fit(ds_train=ds_train, ds_test=ds_test, epochs=EPOCH)
