from voc import load_voc_dataset, prepare, cc2bc
from anchor import SSDAnchorGenerator
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np


def debgd_and_do_offset(offsets, classes, anchor_bboxes):
    # do offsets
    bboxes = np.zeros_like(offsets)
    bboxes[:, 0] = offsets[:, 0] * anchor_bboxes[:, 2] + anchor_bboxes[:, 0]
    bboxes[:, 1] = offsets[:, 1] * anchor_bboxes[:, 3] + anchor_bboxes[:, 1]
    bboxes[:, 2] = np.exp(offsets[:, 2]) * anchor_bboxes[:, 2]
    bboxes[:, 3] = np.exp(offsets[:, 3]) * anchor_bboxes[:, 3]

    # exclude background class
    target_bboxes = []
    for bbox, cls in zip(bboxes, classes):
        if cls[20] == 1:  # class == background
            continue
        target_bboxes.append(bbox[-4:])

    target_bboxes = np.stack(target_bboxes, axis=0)

    return target_bboxes


def draw_img_with_bbox(_ax, img, bboxes):
    """ draw blocks on images
    args:
        _ax: the ax of pyplot
        img: the image
        bboxes: should be boundary coords
    """
    _ax.imshow(img)
    for bbox_coords in bboxes:
        height = img.shape[0]
        width = img.shape[1]
        x_min = bbox_coords[1] * width
        y_min = bbox_coords[0] * height
        x_max = bbox_coords[3] * width
        y_max = bbox_coords[2] * height
        w = x_max - x_min
        h = y_max - y_min

        _ax.add_patch(
            patches.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='red', lw=2))


if __name__ == '__main__':
    ds_train, ds_test = load_voc_dataset()

    print(len(ds_train))

    anchor_gen = SSDAnchorGenerator()
    anchors = anchor_gen.make_anchors_for_multi_fm()

    ds_train = prepare(ds_train)

    n_show = 5

    fig, ax = plt.subplots(1, n_show)
    for i, example in enumerate(ds_train.take(n_show)):

        # test reverse transform
        # bboxes = debgd_and_do_offset(example[1]['target_offsets'].numpy(),
        #                              example[1]['target_classes'].numpy(),
        #                              anchors)
        # draw_img_with_bbox(ax[i], example[0], cc2bc(bboxes))

        # test anchors
        print('anchors center-sized')
        print(anchors)
        draw_img_with_bbox(ax[i], example[0], cc2bc(anchors[:20]))

        # test gt bbox
        # draw_img_with_bbox(ax[i], example['image'], example['objects']['bbox'])

        # print the bbox
        # print(example['objects']['bbox'])

    plt.show()

'''

from modelV2 import SSD, SSDPredConv
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D

class SSDPredConv(Layer):

    def __init__(self, priors_per_tile, fm_size):
        super(SSDPredConv, self).__init__()
        self.priors_per_tile = priors_per_tile
        self.fm_size = fm_size

        self.loc_conv = [Conv2D(priors * 4, (3, 3), padding='same')
                         for priors in self.priors_per_tile]
        self.cls_conv = [Conv2D(priors * 21, (3, 3), padding='same')
                         for priors in self.priors_per_tile]

    def call(self, fms, **kwargs):
        loc_output = []
        cls_output = []
        for i in range(len(fms)):
            loc_tmp = self.loc_conv[i](fms[i])  # H * W * 4
            loc_tmp = tf.reshape(loc_tmp, [-1, self.fm_size[i] * self.priors_per_tile[i], 4])  # reshape
            loc_output.append(loc_tmp)
            cls_tmp = self.cls_conv[i](fms[i])  # H * W * 21
            cls_tmp = tf.reshape(cls_tmp, [-1, self.fm_size[i] * self.priors_per_tile[i], 21])  # reshape
            cls_tmp = tf.math.softmax(cls_tmp, axis=-1)  # softmax to get cls score
            cls_output.append(cls_tmp)
        return {
            'output_loc': tf.convert_to_tensor(tf.concat(loc_output, axis=-2)),
            'output_cls': tf.convert_to_tensor(tf.concat(cls_output, axis=-2))
        }
    

if __name__ == "__main__":
    priors_per_tile = [4, 6, 6, 6, 4, 4]
    fm_size = [38 * 38, 19 * 19, 10 * 10, 5 * 5, 3 * 3, 1 * 1]
    layer = SSDPredConv(priors_per_tile, fm_size)

    # simulate the output from previous aux_conv
    _input = [
        Input((38, 38, 512)),
        Input((19, 19, 1024)),
        Input((10, 10, 512)),
        Input((5, 5, 256)),
        Input((3, 3, 256)),
        Input((1, 1, 256)),
    ]

    _ = layer(_input)

    print(priors_per_tile)

    print(layer.output)

'''
