from voc import load_voc_dataset, prepare

ds_train, ds_test = load_voc_dataset()

ds_train = prepare(ds_train, training=True)

print(ds_train.take(2))

for i in ds_train.take(2):
    print(i)
    break

item = next(iter(ds_train.take(2)))

print(item)