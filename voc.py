import tensorflow_datasets as tfds
import tensorflow as tf
from anchor import SSDAnchorGenerator


# TODO: correct the order of height and width


def load_voc_dataset(sub=True):
    if sub:
        ds_train = tfds.load('voc/2007', split='train+validation', shuffle_files=True)
        ds_val = tfds.load('voc/2007', split='test')
    else:
        ds_train_a = tfds.load('voc/2007', split='train+validation', shuffle_files=True)
        ds_train_b = tfds.load('voc/2012', split='train+validation', shuffle_files=True)
        ds_train = ds_train_b.concat(ds_train_a)
        ds_val = tfds.load('voc/2007', split='test')

    return ds_train, ds_val


def prepare(ds, batch_size, training=False):
    """decode elems in the original dataset and do match
    args:
        the original dataset
    returns:
        a new dataset, each elem is a pair of image and targets
    """

    # TODO: data augmentation and shuffle

    # decode each elem to (image, gts) pair
    ds = ds.map(lambda elem: decode(elem),
                num_parallel_calls=tf.data.AUTOTUNE)

    anchor_gen = SSDAnchorGenerator()
    anchor_bboxes = anchor_gen.make_anchors_for_multi_fm()  # center-sized format

    # data augmentation should be here
    if training:
        pass

    # transform (image, gts) pair to (image, targets)
    ds = ds.map(lambda image, gts: (image, match(cc2bc(anchor_bboxes), gts)),  # convert to boundary coords
                num_parallel_calls=tf.data.AUTOTUNE)

    # shuffle
    ds = ds.batch(batch_size)


    return ds


def decode(elem):
    """ to decode an element from voc dataset provided by tfds
    args:
        elem
            an element from voc dataset provided by tfds
    returns:
        image
            tf.Tensor, the image of the elem
        gt
            a dict, gt['label'] == the label of objects,
            gt['bbox'] == the bouding box of objects, in the
            format of center-size coords
    """
    image = elem['image']
    gts = {
        'bbox': elem['objects']['bbox'],
        'label': elem['objects']['label']
    }

    return image, gts


def cc2bc(cc):
    """convert a group of center-size coords to boundary coords"""
    x_min = cc[:, 0] - cc[:, 2] / 2.
    x_max = x_min + cc[:, 2]
    y_min = cc[:, 1] - cc[:, 3] / 2.
    y_max = y_min + cc[:, 3]

    return tf.stack([x_min, y_min, x_max, y_max], -1)


def bc2cc(bc):
    """convert a group of boundary coords to center-size coords"""
    x_c = (bc[:, 2] + bc[:, 0]) / 2.
    y_c = (bc[:, 3] + bc[:, 1]) / 2.
    w = bc[:, 2] - bc[:, 0]
    h = bc[:, 3] - bc[:, 1]

    return tf.stack([x_c, y_c, w, h], -1)


def cal_iou(d_bboxes, g_bboxes):
    """calculate iou between two groups of bboxes
    args:
        d_bboxes
            a numpy array of bbox coords (with boundary coords)
        g_bboxes
            a numpy array of bbox coords (with boundary coords)
        the two array should have the shape (None, 4), but unnessasary the same
    returns:
        a numpy array of shape (n_objs, n_anchors, 1)
    """

    max_min = tf.math.minimum(g_bboxes[:, :, 2:], d_bboxes[:, :, 2:])
    min_max = tf.math.maximum(g_bboxes[:, :, :2], d_bboxes[:, :, :2])

    mul = (max_min - min_max)
    mul = tf.where(mul > 0, x=mul, y=tf.zeros_like(mul, dtype=tf.float32))

    inter = mul[:, :, 0] * mul[:, :, 1]

    union = (g_bboxes[:, :, 2] - g_bboxes[:, :, 0]) * (g_bboxes[:, :, 3] - g_bboxes[:, :, 1]) + \
            (d_bboxes[:, :, 2] - d_bboxes[:, :, 0]) * (d_bboxes[:, :, 3] - d_bboxes[:, :, 1]) - inter

    iou = inter / union

    return iou


def cal_offset(d_bboxes, g_bboxes):
    """calculate offset from d_bboxes to g_bboxes
    args:
        d_bboxes
            a numpy array of bbox coords (with center-size coords)
        g_bboxes
            a numpy array of bbox coords (with center-size coords)
        the two array should have the same shape (None, 4)
    returns:
        a numpy array of shape (None, 4)
    """

    c_x_offsets = (g_bboxes[:, 0] - d_bboxes[:, 0]) / d_bboxes[:, 2]
    c_y_offsets = (g_bboxes[:, 1] - d_bboxes[:, 1]) / d_bboxes[:, 3]
    w_offsets = tf.math.log(g_bboxes[:, 2] / d_bboxes[:, 2])
    h_offsets = tf.math.log(g_bboxes[:, 3] / d_bboxes[:, 3])

    return tf.stack([c_x_offsets, c_y_offsets, w_offsets, h_offsets], -1)


def match(anchor_bboxes, gts, num_classes_without_bgd=20):
    """map anchors to gts
    args:
        anchor_bbox
            tf.constant, should be boundary coords, with shape [n_anchors, 4]
        gts
            a dict, as defined in the decode function
    returns:
        targets
            [:21] the class label
            [21:] the offsets
    """
    gt_bboxes = gts['bbox']
    gt_labels = gts['label']

    # broadcast to [n_objs, n_anchors, 4]
    n_anchors = tf.shape(anchor_bboxes)[0]
    enlarged_gt_bboxes = tf.repeat(tf.expand_dims(gt_bboxes, 1), n_anchors, 1)  # [n_objs, 4] -> [n_objs, n_anchors, 4]

    n_objs = tf.shape(gt_bboxes)[0]
    enlarged_anchor_bboxes = tf.repeat(tf.expand_dims(anchor_bboxes, 0), n_objs, 0)  # [n_anchors, 4] -> [n_objs, n_anchors, 4]

    ious = cal_iou(enlarged_anchor_bboxes, enlarged_gt_bboxes)  # [n_objs, n_anchors, 1]

    # two rules to do the match depending on ious
    # 1. anchor-wise 2. gt-wise

    # anchor-wise
    max_iou_gt_idxs = tf.math.argmax(ious, axis=0)
    max_iou_gt = tf.math.reduce_max(ious, axis=0)
    target_labels = tf.gather(gt_labels, max_iou_gt_idxs, axis=0)
    target_labels = tf.expand_dims(
        tf.where(max_iou_gt > 0.5, x=target_labels, y=tf.ones_like(target_labels) * num_classes_without_bgd),  # class=21 (background) by default
        -1)
    target_bboxes = tf.gather(gt_bboxes, max_iou_gt_idxs, axis=0)

    anchor_wise_match_results = target_bboxes

    # gt-wise
    gt_idxs = tf.range(0, n_objs, 1, dtype=tf.int64)

    max_iou_anchor_idxs = tf.math.argmax(ious, axis=1)
    for i in tf.range(n_objs):
        tf.autograph.experimental.set_loop_options(  # otherwise error raised
            shape_invariants=[
                (target_bboxes, tf.TensorShape([None, 4])),
                (target_labels, tf.TensorShape([None, 1]))
            ]
        )

        val = max_iou_anchor_idxs[i]  # which anchor
        # for labels
        label_head = target_labels[:val, :]
        label_mid = tf.reshape(gt_idxs[i], [1, 1])
        label_tail = target_labels[val + 1:, :]
        target_labels = tf.concat([label_head, label_mid, label_tail], axis=0)

        # for bboxes
        bbox_head = target_bboxes[:val, :]
        bbox_mid = tf.reshape(gt_bboxes[i], [1, 4])
        bbox_tail = target_bboxes[val + 1:, :]
        target_bboxes = tf.concat([bbox_head, bbox_mid, bbox_tail], axis=0)

    gt_wise_match_results = target_bboxes

    # up to now, all anchors should have a label and a target bbox

    # turn the coords to center-size form
    anchor_bboxes = bc2cc(anchor_bboxes)
    target_bboxes = bc2cc(target_bboxes)

    offsets = cal_offset(anchor_bboxes, target_bboxes)

    # now we have tow tensors, target_labels and offsets,
    # the order is the same as the corresponding anchors'

    # one hot coding
    target_labels = tf.one_hot(
        tf.squeeze(target_labels),
        depth=(num_classes_without_bgd + 1),
        axis=-1,
        dtype=tf.float32)

    targets = tf.concat([target_labels, offsets], axis=-1)

    # return gt_bboxes, anchor_wise_match_results, gt_wise_match_results, offsets, target_bboxes, targets

    return targets
