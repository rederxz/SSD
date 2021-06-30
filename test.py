from VOC import load_voc_dataset, SSDAnchorGenerator, prepare

if __name__ == '__main__':
    ds_train, ds_test = load_voc_dataset()

    # for example in ds_train.take(5):
    #     print(example)

    # anchor_gen = SSDAnchorGenerator()
    # anchors = anchor_gen.make_anchors_for_multi_fm()

    # print(anchors)

    ds_train = prepare(ds_train)

    # for example in ds_train.take(5):
    #     print(example)