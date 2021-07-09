from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

from voc import load_voc_dataset, prepare
from modelV2 import SSD
from loss import SSDLoss


# TODO: add lr schedule
# TODO: add mAP


BATCH_SIZE = 32
EPOCH = 232  # iterations per epoch = 5011 / BATCH_SIZE

# data
ds_train, ds_test = load_voc_dataset()
ds_train = prepare(ds_train, batch_size=BATCH_SIZE, training=True)
ds_test = prepare(ds_test, batch_size=128)

# model, loss, optimizer, metric
model = SSD()
model.build((None, 300, 300, 3))

boundaries = [154, 193]
values = [1e-3, 1e-4, 1e-5]
lr_schedule = PiecewiseConstantDecay(boundaries, values)
optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)

# metrics = [SSDLoss, mAP]
metrics = [SSDLoss, ]

model.compile(loss=SSDLoss, optimizer=optimizer, metrics=metrics)

# train
model.fit(ds_train, validation_data=ds_test, epochs=EPOCH)
