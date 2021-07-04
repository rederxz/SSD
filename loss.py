import tensorflow as tf
from tensorflow.keras.losses import huber  # smooth l1 loss


def hard_negative_mining(outputs_cls, num_classes_without_bgd, max_num):
    # softmax
    loss = tf.math.exp(outputs_cls) / tf.reduce_sum(tf.math.exp(outputs_cls), axis=-1)
    loss = loss[:, num_classes_without_bgd]

    # the hardest top k
    max_neg = tf.math.minimum(max_num, tf.shape(loss)[0])  # get the max amounts of negs
    loss = tf.reduce_sum(tf.math.top_k(loss, max_neg))

    return loss


class SSDLoss:

    def __init__(self, alpha=1., num_classes_without_bgd=20):
        self.alpha = alpha
        self.num_classes_without_bgd = num_classes_without_bgd

    def __call__(self, outputs, targets):
        positive_anchor_idxes = tf.where(targets[:, self.num_classes_without_bgd] != 1)  # positive (or matched) anchors
        num_positive_anchors = tf.shape(positive_anchor_idxes)[0]

        if num_positive_anchors == 0:  # if N == 0, Loss=0
            return 0

        # classification loss
        positive_outputs_cls = outputs[positive_anchor_idxes, :self.num_classes_without_bgd]  # only the cls part
        positive_targets_cls = targets[positive_anchor_idxes, :self.num_classes_without_bgd]  # only the cls part
        positive_loss_cls = tf.math.exp(positive_outputs_cls) / tf.reduce_sum(
            tf.math.exp(positive_outputs_cls),
            axis=-1)
        positive_loss_cls = tf.reduce_sum(positive_loss_cls * positive_targets_cls)

        negative_anchor_idxes = tf.where(targets[:, self.num_classes_without_bgd] == 1)
        negative_outputs_cls = outputs[negative_anchor_idxes, :self.num_classes_without_bgd]  # only the cls part
        # negative_targets_cls = targets[negative_anchor_idxes, :self.num_classes_without_bgd]  # only the cls part
        hard_negative_loss_cls = hard_negative_mining(negative_outputs_cls,
                                                      self.num_classes_without_bgd,
                                                      3*num_positive_anchors)  # positive : negative >= 1 :  3

        loss_cls = positive_loss_cls + hard_negative_loss_cls

        # location loss
        outputs_loc = outputs[positive_anchor_idxes, self.num_classes_without_bgd:]  # select location part
        targets_loc = targets[positive_anchor_idxes, self.num_classes_without_bgd:]
        h = huber()
        loss_loc = h(targets_loc, outputs_loc)

        loss = (loss_cls + self.alpha * loss_loc) / num_positive_anchors

        return loss
