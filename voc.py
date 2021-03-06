import tensorflow_datasets as tfds
import tensorflow as tf
from anchor import SSDAnchorGenerator


def cc2bc(cc):
    """convert a group of center-size coords to boundary coords"""
    y_min = cc[:, 0] - cc[:, 2] / 2.
    y_max = y_min + cc[:, 2]
    x_min = cc[:, 1] - cc[:, 3] / 2.
    x_max = x_min + cc[:, 3]

    return tf.stack([y_min, x_min, y_max, x_max], -1)


def bc2cc(bc):
    """convert a group of boundary coords to center-size coords"""
    y_c = (bc[:, 2] + bc[:, 0]) / 2.
    x_c = (bc[:, 3] + bc[:, 1]) / 2.
    h = bc[:, 2] - bc[:, 0]
    w = bc[:, 3] - bc[:, 1]

    return tf.stack([y_c, x_c, h, w], -1)


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

    c_y_offsets = (g_bboxes[:, 0] - d_bboxes[:, 0]) / d_bboxes[:, 2]
    c_x_offsets = (g_bboxes[:, 1] - d_bboxes[:, 1]) / d_bboxes[:, 3]
    h_offsets = tf.math.log(g_bboxes[:, 2] / d_bboxes[:, 2])
    w_offsets = tf.math.log(g_bboxes[:, 3] / d_bboxes[:, 3])

    return tf.stack([c_y_offsets, c_x_offsets, h_offsets, w_offsets], -1)


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


def decode(elem, res=300):
    """ to decode an element from voc dataset provided by tfds to (image, gts) pair
    args:
        elem
            an element from voc dataset provided by tfds
    returns:
        image
            tf.Tensor, the image of the elem
        gts
            tf.Tensor, shape [n_obj, 4 + 1], center-sized bbox
    """
    image = tf.image.resize(elem['image'], (res, res))

    gts_bboxes = elem['objects']['bbox']
    # print('gts_bboxes shape')
    # print(gts_bboxes.shape)

    gts_labels = tf.cast(elem['objects']['label'][..., None], dtype=tf.float32)
    # print('gts_labels shape')
    # print(gts_labels.shape)

    gts = tf.concat([gts_bboxes, gts_labels], axis=-1)
    # print('gts shape')
    # print(gts.shape)

    return image, gts


def match(anchor_bboxes, gts, num_classes_without_bgd=20):
    """map anchors to gts
    args:
        anchor_bbox
            tf.constant, should be boundary coords, with shape [n_anchors, 4]
        gts
            as defined in the decode function (with boundary coords)
    returns:
        targets
            [:21] the class label
            [21:] the offsets
    """
    # gt_bboxes = gts['bbox']
    # gt_labels = gts['label']

    gt_bboxes = gts[..., :4]
    gt_labels = gts[..., -1]

    # broadcast to [n_objs, n_anchors, 4] and calculate IoU
    n_anchors = anchor_bboxes.shape[0]
    enlarged_gt_bboxes = tf.repeat(tf.expand_dims(gt_bboxes, 1), n_anchors, 1)  # [n_objs, 4] -> [n_objs, n_anchors, 4]
    n_objs = tf.shape(gt_bboxes)[0]
    enlarged_anchor_bboxes = tf.repeat(tf.expand_dims(anchor_bboxes, 0), n_objs, 0)  # [n_anchors, 4] -> [n_objs, n_anchors, 4]

    ious = cal_iou(enlarged_anchor_bboxes, enlarged_gt_bboxes)  # [n_objs, n_anchors, 1]

    # two rules to do label alignment from IoU
    # 1. IoU > 0.5
    max_iou_gt_idxs = tf.math.argmax(ious, axis=0)
    max_iou_gt = tf.math.reduce_max(ious, axis=0)
    target_labels = tf.gather(gt_labels, max_iou_gt_idxs, axis=0)
    target_labels = tf.expand_dims(
        tf.where(max_iou_gt > 0.5, x=target_labels, y=tf.ones_like(target_labels) * num_classes_without_bgd),  # class=21 (background) by default
        -1)
    target_bboxes = tf.gather(gt_bboxes, max_iou_gt_idxs, axis=0)

    # 2. best matched bboxes wrt gts
    gt_idxs = tf.range(0, n_objs, 1, dtype=tf.float32)
    max_iou_anchor_idxs = tf.math.argmax(ious, axis=1)
    for i in tf.range(n_objs):
        val = max_iou_anchor_idxs[i]  # which anchor
        # for labels
        label_head = target_labels[:val, :]
        label_mid = tf.reshape(gt_idxs[i], [1, 1])
        label_tail = target_labels[val + 1:, :]
        target_labels = tf.concat([label_head, label_mid, label_tail], axis=0)
        target_labels = tf.reshape(target_labels, (n_anchors, 1))

        # for bboxes
        bbox_head = target_bboxes[:val, :]
        bbox_mid = tf.reshape(gt_bboxes[i], [1, 4])
        bbox_tail = target_bboxes[val + 1:, :]
        target_bboxes = tf.concat([bbox_head, bbox_mid, bbox_tail], axis=0)
        target_bboxes = tf.reshape(target_bboxes, (n_anchors, 4))

    # up to now, all anchors should have a label and a target bbox

    # turn the coords to center-size form and calculate offsets
    anchor_bboxes = bc2cc(anchor_bboxes)
    target_bboxes = bc2cc(target_bboxes)
    target_offsets = cal_offset(anchor_bboxes, target_bboxes)

    # now we have two tensors, target_labels and offsets,
    # the order is the same as the corresponding anchors'

    # print('target_labels shape')
    # print(target_labels.shape)

    # one hot coding
    target_labels = tf.one_hot(
        tf.cast(tf.squeeze(target_labels), dtype=tf.int32),
        depth=(num_classes_without_bgd + 1),
        axis=-1,
        dtype=tf.float32)

    # print('target_labels shape')
    # print(target_labels.shape)

    return tf.concat([target_offsets, target_labels], axis=-1)


def prepare(ds, res=300, batch_size=32, training=False):
    """decode elems in the original dataset and do match
    args:
        the original dataset
    returns:
        a new dataset, each elem is a pair of image and targets
    """

    # FIXME: do normalization

    # decode each elem to (image, gts) pair, and resize
    ds = ds.map(lambda elem: decode(elem, res))

    # data augmentation should be here
    if training:
        # TODO: data augmentation
        pass

    anchor_gen = SSDAnchorGenerator()
    anchor_bboxes = anchor_gen.make_anchors_for_multi_fm()  # center-sized format

    # transform (image, gts) pair to (image, targets, gts)
    ds = ds.map(lambda image, gts: (
        image,
        {'targets': match(cc2bc(anchor_bboxes), gts),  # to calculate loss
         'gts': gts}  # to calculate mAP
    ))

    # shuffle
    ds = ds.shuffle(buffer_size=len(ds))

    # batch
    # ds = ds.padded_batch(batch_size, padding_values=(0., (0., 0., {'bbox': 0., 'label': 20.})), drop_remainder=True)
    ds = ds.padded_batch(batch_size, padding_values=(0., {'targets': -1., 'gts': -1.}), drop_remainder=True)

    return ds

