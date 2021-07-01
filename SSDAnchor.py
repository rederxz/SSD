import math

import tensorflow as tf


"""TODO:
    1. make sure the order of each anchor whether correspond
    to the order of flattened feature map, for func
        make_anchors_for_one_fm
        make_anchors_for_multi_fm
    2. for SSDAnchorGenerator, in case we want to change the default
    settings
"""


class SSDAnchorGenerator:
    """generate anchors defined in SSD"""

    def __init__(self, default=True):
        if default:
            # for each feature map
            self.fm_scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
            self.fm_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]  # (height, width)
            self.fm_subset = [True, False, False, False, True, True]

            # for each tile in a feature map
            self.bbox_aspects = [1., 2., 0.5, 3., 1. / 3.]
        else:
            pass

    def make_anchors_for_one_fm(self, size, scale, extra_scale, subset):
        """make anchors for a particular feature map
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
        """
        anchors = []

        # width and height wrt each aspect
        aspects = self.bbox_aspects[:3] if subset else self.bbox_aspects
        w = [scale * math.sqrt(a) for a in aspects] + [extra_scale]
        h = [scale / math.sqrt(a) for a in aspects] + [extra_scale]
        w_h_pairs = zip(w, h)

        # the order matters, 'i' and 'j' (tiles in the feature map) must be iterated after the aspects,
        # to ensure that neighbour anchors have the same aspect
        for w_h in w_h_pairs:
            for i in range(size[0]):  # size is in the order of (H, W), height direction
                for j in range(size[1]):  # size is in the order of (H, W), width direction
                    c_x_y = [(i + 0.5) / size[1], (j + 0.5) / size[0]]
                    anchors.append(c_x_y + list(w_h))  # [x_c, y_c, w, h]
        return anchors

    def make_anchors_for_multi_fm(self):
        """make anchors for multiple feature maps in different stages
        returns:
            a list of dict
                each dict['bbox'] is the coords of the anchor, in the order of different stage, setting,
                width, height
        """
        anchors = []

        # add 1.0 in the end to calculate extra scales
        scales = self.fm_scales + [1.0]

        for i in range(len(self.fm_scales)):
            extra_scale = math.sqrt(scales[i] * scales[i + 1])  # the extra scale for each stage of feature map
            anchors += self.make_anchors_for_one_fm(self.fm_sizes[i], self.fm_scales[i], extra_scale, self.fm_subset[i])

        anchors = tf.constant(anchors)

        return anchors
