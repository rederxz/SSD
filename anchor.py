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

    def __init__(self, fm_scales=None, fm_sizes=None, fm_subset=None):
        # for each feature map
        if fm_scales is None:
            self.fm_scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        if fm_sizes is None:
            self.fm_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]  # (height, width)
        if fm_subset is None:
            self.fm_subset = [True, False, False, False, True, True]

        # for each tile in a feature map
        self.bbox_aspects = [1., 2., 0.5, 3., 1. / 3.]

    def make_anchors_for_one_fm(self, size, scale, extra_scale, subset):
        """make anchors for a particular feature map
        the order of traverse: y direction -> x direction -> different aspect ratios
        [
            the position of (y_0, x_0)

            [y_0, x_0, h_0, w_0], -> group 0 of aspects ratio
            [y_0, x_0, h_1, w_1], -> group 1 ~
            [y_0, x_0, h_2, w_2], -> group 2 ~
            [y_0, x_0, h_3, w_3], -> group 3 ~
            ...                   -> until the last group

            the position of (y_0, x_1)

            [y_0, x_1, h_0, w_0],
            [y_0, x_1, h_1, w_1],
            [y_0, x_1, h_2, w_2],
            [y_0, x_1, h_3, w_3],
            ...

            ...

            cross the horizontal line of tiles on the feature map, till:

            [y_0, x_n, h_0, w_0],
            [y_0, x_n, h_1, w_1],
            [y_0, x_n, h_2, w_2],
            [y_0, x_n, h_3, w_3],
            ...

            then start the next horizontal line

            the position of (y_1, x_0)
            [y_1, x_0, h_0, w_0],
            [y_1, x_0, h_1, w_1],
            [y_1, x_0, h_2, w_2],
            [y_1, x_0, h_3, w_3],
            ...

            ...
        ]
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
                each elem is the coords of the anchor, in the form of (y, x, height, width)
        """
        anchors = []

        # width and height wrt each aspect
        aspects = self.bbox_aspects[:3] if subset else self.bbox_aspects
        h = [scale / math.sqrt(a) for a in aspects] + [extra_scale]
        w = [scale * math.sqrt(a) for a in aspects] + [extra_scale]

        # the order matters, 'i' and 'j' (tiles in the feature map) must be iterated after the aspects,
        # to ensure that neighbour anchors have the same aspect
        for i in range(size[0]):  # size is in the order of (H, W), height direction
            for j in range(size[1]):  # size is in the order of (H, W), width direction
                c_y_x = [(i + 0.5) / size[0], (j + 0.5) / size[1]]
                for h_w in zip(h, w):
                    anchors.append(c_y_x + list(h_w))  # [y_c, x_c, h, w]
        return anchors

    def make_anchors_for_multi_fm(self):
        """make anchors for multiple feature maps in different stages
        """
        anchors = []

        # add 1.0 in the end to calculate extra scales
        scales = self.fm_scales + [1.0]

        for i in range(len(self.fm_scales)):
            extra_scale = math.sqrt(scales[i] * scales[i + 1])  # the extra scale for each stage of feature map
            anchors += self.make_anchors_for_one_fm(self.fm_sizes[i], self.fm_scales[i], extra_scale, self.fm_subset[i])

        return tf.constant(anchors)

