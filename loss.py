import tensorflow as tf
from tensorflow.keras.losses import huber  # smooth l1 loss


def hard_negative_mining(outputs_cls, max_num):
    # get the softmax of the background class
    loss = -1 * tf.math.log(outputs_cls[:, -1])

    # the hardest top k
    max_neg = tf.math.minimum(max_num, tf.shape(loss)[0])  # get the max amounts of negs
    return tf.reduce_sum(tf.math.top_k(loss, max_neg))


class SSDLoss:
    """Notion: the last elem of the output of the classification should be background
    """

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        print(y_pred)
        target_cls = y_true['classes']
        target_offset = y_true['offsets']

        output_cls = y_pred['classes']
        output_offset = y_pred['offsets']

        positive_anchor_idxes = tf.where(target_cls[:, -1] != 1)  # positive (or matched) anchors
        num_positive_anchors = tf.shape(positive_anchor_idxes)[0]

        if num_positive_anchors == 0:  # if N == 0, Loss=0
            return 0

        # classification loss
        # positive
        positive_outputs_cls = output_cls[positive_anchor_idxes]
        positive_targets_cls = target_cls[positive_anchor_idxes]
        positive_loss_cls = -1 * tf.reduce_sum(tf.math.log(positive_outputs_cls * positive_targets_cls))

        # negative
        negative_anchor_idxes = tf.where(target_cls[:, -1] == 1)
        negative_outputs_cls = output_cls[negative_anchor_idxes]
        hard_negative_loss_cls = hard_negative_mining(negative_outputs_cls,
                                                      3*num_positive_anchors)  # positive : negative >= 1 :  3

        loss_cls = positive_loss_cls + hard_negative_loss_cls

        # location loss
        outputs_loc = output_offset[positive_anchor_idxes]
        targets_loc = target_offset[positive_anchor_idxes]
        loss_loc = huber()(targets_loc, outputs_loc)

        loss = (loss_cls + self.alpha * loss_loc) / num_positive_anchors

        return loss
