from voc import load_voc_dataset, SSDAnchorGenerator, prepare, cc2bc
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np


def debgd_and_do_offset(bboxes, anchor_bboxes):
    # do offsets
    offsets = bboxes[:, -4:]
    bboxes[:, -4] = offsets[:, 0] * anchor_bboxes[:, 2] + anchor_bboxes[:, 0]
    bboxes[:, -3] = offsets[:, 1] * anchor_bboxes[:, 3] + anchor_bboxes[:, 1]
    bboxes[:, -2] = np.exp(offsets[:, 2]) * anchor_bboxes[:, 2]
    bboxes[:, -1] = np.exp(offsets[:, 3]) * anchor_bboxes[:, 3]

    # exclude background class
    target_bboxes = []
    for bbox in bboxes:
        if bbox[20] == 1:  # class == background
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
        # [1][-2] target_bboxes
        # [1][-3] offsets
        # [1][1] anchorwise match result
        # [1][2] gtwise match result

        # test reverse transform
        # bboxes = debgd_and_do_offset(example[1][-1].numpy(), anchors)
        # draw_img_with_bbox(ax[i], example[0], cc2bc(bboxes))

        # test anchors
        # print('anchors center-sized')
        # print(anchors)
        draw_img_with_bbox(ax[i], example[0], cc2bc(anchors[:20]))

        # test target coords
        # draw_img_with_bbox(ax[i], example[0], cc2bc(example[1][-2][7000:7020]))# test_targets
        # print('before convert')
        # print(example[1][2])
        # print('target coords after convert')
        # print(example[1][-2])

        # test gt_bboxes
        # draw_img_with_bbox(ax[i], example[0], example[1][0])

        # test offsets
        # print('offsets')
        # print(example[1][-3])


    plt.show()

    # for example in ds_train.take(5):
    #     print(example)
