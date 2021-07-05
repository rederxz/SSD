from voc import load_voc_dataset, SSDAnchorGenerator, prepare, cc2bc
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

    anchor_gen = SSDAnchorGenerator()
    anchors = anchor_gen.make_anchors_for_multi_fm()

    ds_train = prepare(ds_train)

    n_show = 5

    fig, ax = plt.subplots(1, n_show)
    for i, example in enumerate(ds_train.take(n_show)):

        # test reverse transform
        bboxes = debgd_and_do_offset(example[1]['target_offsets'].numpy(),
                                     example[1]['target_classes'].numpy(),
                                     anchors)
        draw_img_with_bbox(ax[i], example[0], cc2bc(bboxes))

        # test anchors
        # print('anchors center-sized')
        # print(anchors)
        # draw_img_with_bbox(ax[i], example[0], cc2bc(anchors[:20]))

        # test gt bbox
        # draw_img_with_bbox(ax[i], example['image'], example['objects']['bbox'])

        # print the bbox
        # print(example['objects']['bbox'])

    plt.show()




