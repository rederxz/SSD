import tensorflow as tf
import tensorflow.keras.backend as K


def nms(bboxes, scores, iou_threshold=0.45, score_threshold=0.01, top_k=200, max_per_class=200):
    """do non-maximum suppression with given iou_limit and score_limit
    args:
        bboxes: anchor bboxes from an image, in boundary size form
        scores: anchor scores of each class from an image
        iou_limit: to define the redundant bboxes in nms
        score_limit: the threshold to consider a bbox is valid
    returns:
        labels: tensor with shape (#objs, #classes)
        scores: tensor with shape (#objs, )
        bboxes: tensor with shape (#objs, 4), boundary coords
        n_valid: scalar, number of valid objects in the image
    """
    # exclude bboxes with too low score
    max_scores = tf.reduce_max(scores, axis=-1)
    idx = tf.where(max_scores >= score_threshold)
    scores = scores[idx, ...]  # [n_bboxes after selection, n_classes]
    bboxes = bboxes[idx, ...]  # [n_bboxes after selection, 4]

    # do nms for each class
    n_bboxes, n_classes = tf.shape(scores)

    all_labels = []
    all_scores = []
    all_bboxes = []
    all_n_valid = []
    for _class in range(n_classes):
        scores_wrt_class = scores[:, _class]
        idx, scores_cls, n_valid = tf.raw_ops.NonMaxSuppressionV5(
                boxes=bboxes, scores=scores_wrt_class, max_output_size=max_per_class,
                iou_threshold=iou_threshold, score_threshold=score_threshold,
                soft_nms_sigma=0., pad_to_max_output_size=False)
        all_scores.append(scores_cls)
        all_labels.append(tf.ones_like(idx) * _class)
        all_bboxes.append(tf.gather(bboxes, idx))
        all_n_valid.append(n_valid)

    all_labels = tf.stack(all_labels, axis=0)
    all_scores = tf.stack(all_scores, axis=0)
    all_bboxes = tf.stack(all_bboxes, axis=0)
    all_n_valid = tf.reduce_sum(tf.stack(all_n_valid))

    if all_n_valid > top_k:
        all_scores, idx = tf.math.top_k(all_scores, top_k=top_k, sorted=True)
        all_labels = tf.gather(all_labels, idx)
        all_bboxes = tf.gather(all_bboxes, idx)
        all_n_valid = top_k

    return all_labels, all_scores, all_bboxes, all_n_valid


def batched_nms(bboxes, scores, iou_limit=0.5, score_limit=0.02):
    batch_size = tf.shape(bboxes)[0]
    output = []
    for i in range(batch_size):
        output.append(nms(bboxes[i], scores[i], iou_limit, score_limit))
    return [tf.stack(_) for _ in zip(*output)]  # [labels, scores, bboxes, n_valid]


def do_offsets(offsets, anchor_bboxes):
    # do offsets
    bboxes = K.zeros_like(offsets)
    bboxes[..., 0] = offsets[..., 0] * anchor_bboxes[..., 2] + anchor_bboxes[..., 0]
    bboxes[..., 1] = offsets[..., 1] * anchor_bboxes[..., 3] + anchor_bboxes[..., 1]
    bboxes[..., 2] = tf.math.exp(offsets[..., 2]) * anchor_bboxes[..., 2]
    bboxes[..., 3] = tf.math.exp(offsets[..., 3]) * anchor_bboxes[..., 3]
    return bboxes


def postprocess(outputs, anchors):
    """
    postprocess the output of SSD model, including:
        1. do the offsets and convert to boundary coords
        2. None maximum suppression;
    args:
        outputs: the output of the SSD model
        anchors: anchors as defined
    returns:
        four lists: labels, scores, bboxes, n_valids of each image
    """

    bboxes = do_offsets(outputs['offsets'], anchors)
    labels, scores, bboxes, n_valid = batched_nms(bboxes, outputs['classes'])

    return {
        'labels': labels,
        'scores': scores,
        'bboxes': bboxes,
        'n_valid': n_valid
    }


def mAP(targets, outputs):
    """we need to consider that: mAP needs the entire valuation dataset, but the evaluation is performed batch by batch,
    how to update the
    """
    raise NotImplementedError
