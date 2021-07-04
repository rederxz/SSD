# TODO:


def postprocess(outputs, anchors):
    """
    postprocess the output of SSD model, including:
        1. apply threshold to the classification scores;
        2. None maximum suppression;
    args:
        outputs: the output of the SSD model
        anchors: anchors as defined
    returns:
        precision: precision of the result
    """
    raise NotImplementedError


def cal_mAP():
    raise NotImplementedError
