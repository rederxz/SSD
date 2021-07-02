from VOC import load_voc_dataset, prepare
from Model import  SSD300
from SSDLoss import SSDLoss
from tensorflow.keras.optimizers import SGD


BATCH_SIZE=32


ds_train, ds_test = load_voc_dataset()
ds_train = prepare(ds_train, batch_size=BATCH_SIZE, training=True)
ds_test = prepare(ds_test, batch_size=128)

model = SSD300()
optimizer = SGD(learning_rate=1e-3, momentum=0.9)
model.compile(loss=SSDLoss(), optimizer=optimizer)

model.fit(ds_train=ds_train, ds_test=ds_test)
