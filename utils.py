def postprocess(outputs, anchors):
    """
    postprocess the output of SSD model, including:
        apply threshold to the classification scores;
        None maximum suppression;
    args:
        outputs: the output of the SSD model
        anchors: anchors as defined
    returns:
        precision: precision of the result
    """