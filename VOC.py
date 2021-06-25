from re import I
import tensorflow_datasets as tfds
import tensorflow as tf
# import numpy as np
import math


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

'''TODO:
    add data augmentation
'''
def prepare(ds):
    '''decode elems in the original dataset and do match
    args:
        the original dataset
    returns:
        a new dataset, each elem is a pair of image and targets
    '''
    
    # decode each elem to (image, gts) pair
    ds = ds.map(lambda elem: decode(elem),
                num_parallel_calls=tf.data.AUTOTUNE)

    anchor_gen = SSDAnchorGenerator()
    anchor_bboxes = anchor_gen.make_anchors_for_multi_fm()

    # decode (image, gts) pair to (image, targets)
    ds = ds.map(lambda image, gts : (image, match(anchor_bboxes, gts)),
                num_parallel_calls=tf.data.AUTOTUNE)

    return ds


'''TODO:
    1. check where to add the 'bgd' class - might be in the match function
    
'''


def decode(elem):
    ''' to decode an element from voc dataset provided by tfds
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
    '''
    image = elem['image']
    gts = {
        'bbox': elem['objects']['bbox'],
        'label': elem['objects']['label']
    }

    return image, gts


def cc2bc(cc):
    '''convert a group of center-size coords to boundary coords'''
    x_min = cc[:, 0] - cc[:, 2] / 2.
    x_max = x_min + cc[:, 2]
    y_min = cc[:, 1] - cc[:, 3] / 2.
    y_max = y_min + cc[:, 3]
    
    return tf.stack([x_min, y_min, x_max, y_max], -1)

def bc2cc(bc):
    '''convert a group of coords to center-size coords'''
    x_c = (bc[:, 2] - bc[:, 0]) / 2.
    y_c = (bc[:, 3] - bc[:, 1]) / 2.
    w = bc[:, 2] - bc[:, 0]
    h = bc[:, 3] - bc[:, 1]
    
    return tf.stack([x_c, y_c, w, h], -1)

def cal_iou(d_bboxes, g_bboxes):
    '''calculate iou betwwen two groups of bboxes
    args:
        d_bboxes
            a numpy array of bbox coords (with boundary coords)
        g_bboxes
            a numpy array of bbox coords (with boundary coords)
        the two array should have the shape (None, 4), but unnessasary the same
    returns:
        a numpy array of shape (n_objs, n_anchors, 1)
    '''

    max_min = tf.math.minimum(g_bboxes[:, :, 2:], d_bboxes[:, :, 2:])
    min_max = tf.math.maximum(g_bboxes[:, :, :2], d_bboxes[:, :, :2])

    mul = (max_min - min_max)
    mul = mul * tf.where(mul > 0, x=tf.ones_like(mul, dtype=tf.float32),
                    y=tf.zeros_like(mul, dtype=tf.float32))
    inter = mul[:, :, 0] * mul[:, :, 1]

    union = (g_bboxes[:, :, 2] - g_bboxes[:, :, 0]) * (g_bboxes[:, :, 3] - g_bboxes[:, :, 1]) + \
            (d_bboxes[:, :, 2] - d_bboxes[:, :, 0]) * (d_bboxes[:, :, 3] - d_bboxes[:, :, 1]) - inter

    iou = inter / union
    
    return iou

def cal_offset(d_bboxes, g_bboxes):
    '''calculate offset from d_bboxes to g_bboxes
    args:
        bboxes_a
            a numpy array of bbox coords (with center-size coords)
        bboxes_b
            a numpy array of bbox coords (with center-size coords)
        the two array should have the same shape (None, 4)
    returns:
        a numpy array of shape (None, 4)
    '''

    c_x_offsets = (g_bboxes[:, 0] - d_bboxes[:, 0]) / d_bboxes[:, 0]
    c_y_offsets = (g_bboxes[:, 1] - d_bboxes[:, 1]) / d_bboxes[:, 1]
    w_offsets = tf.math.log(g_bboxes[:, 2] / d_bboxes[:, 2])
    h_offsets = tf.math.log(g_bboxes[:, 3] / d_bboxes[:, 3])

    return tf.stack([c_x_offsets, c_y_offsets, w_offsets, h_offsets], -1)

def match(anchor_bboxes, gts, num_classes_without_bgd=20):
    '''map anchors to gts
    args:
        anchor_bbox
            tf.constant, as defined in the SSDAnchorGenerator, with shape [n_anchors, 4]
        gts
            a dict, as defined in the decode function
    returns:
        targets
            a list of target,  for each target
            target['label'] is the target class
            target['offset'] is [g_c_x, g_c_y, g_w, g_h]
    '''
    gt_bboxes = gts['bbox']
    gt_labels = gts['label']

    # broadcast to [n_objs, n_anchors, 4]
    n_anchors = tf.shape(anchor_bboxes)[0]
    enlarged_gt_bboxes = tf.repeat(tf.expand_dims(gt_bboxes, 1), n_anchors, 1) # [n_objs, 4] -> [n_objs, n_anchors, 4]

    # enlarged_anchor_bboxes = tf.ones_like(enlarged_gt_bboxes)
    n_objs = tf.shape(gt_bboxes)[0]
    enlarged_anchor_bboxes = tf.repeat(tf.expand_dims(anchor_bboxes, 1), n_objs, 0) # [n_anchors, 4] -> [n_objs, n_anchors, 4]

    ious = cal_iou(enlarged_anchor_bboxes, enlarged_gt_bboxes) # [n_objs, n_anchors, 1]

    # two rules to do the match depending on ious
    # 1. anchor-wise 2. gt-wise

    # anchor-wise
    max_iou_gt_idxs = tf.math.argmax(ious, axis=0)
    max_iou_gt = tf.math.reduce_max(ious, axis=0)
    target_labels = tf.gather(gt_labels, max_iou_gt_idxs)
    target_labels = tf.expand_dims(tf.where(max_iou_gt > 0.5,
                    x=target_labels, y=tf.ones_like(target_labels) * num_classes_without_bgd), -1)
    target_bboxes = tf.gather(gt_bboxes, max_iou_gt_idxs)

    print(target_labels)
    print(target_bboxes)
    
    # gt-wise
    gt_idxs = tf.range(0, n_objs, 1, dtype=tf.int64)
    max_iou_anchor_idxs = tf.math.argmax(ious, axis=1)
    for i in range(n_objs):
        val = max_iou_anchor_idxs[i] # which anchor
        # fot labels
        label_head = target_labels[:val, :]
        label_mid = tf.reshape(gt_idxs[i], [1, 1])
        label_tail = target_labels[val+1:, :]
        target_labels = tf.concat([label_head, label_mid, label_tail], axis=0)

        # for bboxes
        bbox_head = target_bboxes[:val, :]
        bbox_mid = tf.expand_dims(gt_bboxes[i], 0)
        bbox_tail = target_bboxes[val+1:, :]
        target_bboxes = tf.concat([bbox_head, bbox_mid, bbox_tail], axis=0)

    # up to now, all anchors should have a label and a target bbox

    print(target_labels)
    print(target_bboxes)
    print(anchor_bboxes)

    # turn the coords to center-size form
    anchor_bboxes = bc2cc(anchor_bboxes)
    target_bboxes = bc2cc(target_bboxes)

    offsets = cal_offset(anchor_bboxes, target_bboxes)

    print(offsets)

    # now we have tow tensors, target_labels and offsets, 
    # the order is the same as the corresponding anchors'

    # one hot 
    target_labels = tf.one_hot(
        tf.squeeze(target_labels), 
        depth=(num_classes_without_bgd+1), 
        axis=-1,
        dtype=tf.float32)   

    print(target_labels)

    targets = tf.concat([target_labels, offsets], axis=-1)

    print(targets)
    
    return targets

'''TODO:
    1. make sure the order of each anchor whether correspond
    to the order of flattened feature map, for func
        make_anchors_for_one_fm
        make_anchors_for_multi_fm
    2. for SSDAnchorGenerator, in case we want to change the default
    settings
'''
class SSDAnchorGenerator:
    ''' generate anchors defined in SSD'''
    
    def __init__(self, default=True):
        if default:
            # for each feature map
            self.fm_scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
            self.fm_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
            self.fm_subset = [True, False, False, False, True, True]

            # for each tile in a feature map
            self.bbox_aspects = [1., 2., 0.5, 3., 1./3.]
        else:
            pass    

    def make_anchors_for_one_fm(self, size, scale, extra_scale, subset):
        ''' make anchors for a particular feature map
        args:
            size
                tuple (H, W), hight and width of the feature map
            scale
                float, the scale of each anchor, the area in the image is scale * scale
            extra_scale
                float, the extra scale for each anchor with aspect == 1:1
            subset
                bool, whether to use the simple aspects
        returns:
            a list of tf.constant
                each elem is the coords of the anchor,  anchors are 
                in the order of different setting, width, height
        '''
        anchors = []

         # width and height wrt each aspect
        aspects = self.bbox_aspects[:3] if subset else self.bbox_aspects
        w = [scale * math.sqrt(a) for a in aspects] + [extra_scale]
        h = [scale / math.sqrt(a) for a in aspects] + [extra_scale]
        w_h_pairs = zip(w, h)

        # the order matters, 'i' and 'j' (tiles in the feature map) must be iterated after the aspects,
        # to ensure that neighbour anchors have the same aspect
        for w_h in w_h_pairs:
            for j in range(size[0]): # size is in the order of (H, W), height direction
                for i in range(size[1]): # size is in the order of (H, W), width direction
                    c_x_y = [(i + 0.5) / size[1], (j + 0.5) / size[0]] 
                    anchors.append(c_x_y + list(w_h)) # [x_c, y_c, w, h]
        return anchors

    def make_anchors_for_multi_fm(self):
        '''make anchors for multiple feature maps in different stages
        returns:
            a list of dict
                each dict['bbox'] is the coords of the anchor, in the order of different stage, setting, 
                width, height
        '''
        anchors = []

        # add 1.0 in the end to calculate extra scales
        scales = self.fm_scales + [1.0]

        for i in range(len(self.fm_scales)):
            extra_scale = math.sqrt(scales[i] * scales[i+1]) # the extra scale for each stage of feature map
            anchors += self.make_anchors_for_one_fm(self.fm_sizes[i], self.fm_scales[i], extra_scale, self.fm_subset[i])

        anchors = tf.constant(anchors)

        return anchors
