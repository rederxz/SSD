SSD single short object detector

basic concepts:

- box, bounding box (contains an objec)
- boundary coordinates (x_min, x_max, y_min, y_max), center size coordinates (x_c, y_c, width, height)
- jaccard index 交并比
  - 1 means identiy, 0 means exclusive
- multibox
  - return 2 things, corordinates and class-specifc score (plus background class)

rodeo

- priors
  - some box manually assigned across the image
  - match patches on feature maps of  some base and aux conv layers
  - they are **fixed**
- base conv,  auxiliary conv,  prediction conv
  - base conv, adjust
  - auxiliary conv, to get multi scale info (understand how different stages match different region in the images)
  - prediction conv, loc & cls
- how to match predictions (bounding box and class score) to the GT
  - rules of 'postive' and 'negative' priors
    - jaccard index (J) < 0.5 with any GT, negative
    - J > 0.5 with a GT, positive
    - **what if J > 0.5 with more than one object?**
  - find the targets: predictions - priors - GT
    - target for **loc**: positive - GT's coordinate; negative: dropped
    - tartget for **cls**: positive - GT's class; negative: backgroud class hard negative mining
- dealing with the redundant predictions (during test time)
  - none maximum suppression

implementation (SSD300)

- base-conv

  - load pretrained model in keras (vgg16 including top)

  - a new model, which: 

    (4 modifications)

    - converts the fc_6 and fc_7 to conv layer, and drop fc_8
    - modify the input shape to (300, 300, 3)
    - 3rd pooling layer's output size
    - 5th pooling layer from 2x2_s2 to 3x3_s1

- aux-conv

  - add 4 more layers (each contains 2 conv layers, downsampling as going)

- predictiion conv

  - from 6 feature maps, total 8732 priors
  - for each feature map (N priors per tile, C+1 classes)
    - assign **N\*(C+1)** kernels to classify
    - assign **N\*4** kernels to locate
  - 3x3 kernel is used, padding='same'
  - output is **8732 \* (C+1+4)**

- matching the prediction output to GT

  - calculate the jarccard index between the 8732 priors and **N** GT objects, for each prior, get the max jarccard index

  - maximum > 0.5, positive match with the object

    **(what if match more than one object?)**

  - otherwise, negative match

  - then we got prediction-prior-object match

  - for each match, we have

    - **type**: postive or negative
    - **class**: bgd if negative else coresponding class
    - **coordinate**: GT coordinate if positive else none

- loss

  - loss - cls
    - hard negative mininig, positive : negative = 3 : 1
    - averaged by num_positive
  - loss - loc vanilla
  - sum

- test time

  - for each none-bgd class:
    - for each candidate
      - eliminate boxes that do not meet a certain threshold for this score
    - rank by higher likelihood, eliminate all candidates that have a jaccard overlap with current candidate of more than a threshold, until the end

tour

- voc 数据集下载地址

  - https://pjreddie.com/projects/pascal-voc-dataset-mirror/
  - https://hyper.ai/datasets/7660

- 目标检测数据集下载 https://www.cnblogs.com/zi-wang/p/12325664.html

- 最后使用tfds.load， for each example：

  - 'image' (original shape) uint8
  - 'objects' 'bbox' (N, 4) float32
  - 'objects' 'is_difficult' (N,) bool
  - 'objects' 'label' (N,) int64

- One of the most tough thing to do is to prepare the dataset. We need to:

  - **get the GT bounding boxes of each image, and create a label to match the output of SSD according to the bounding boxes. (We need to generate sth of a certain shape (probably like (8732, ...))  for each image)**
  - generate a marix with shape == (8732, 4) to represent the location of the 8732 priors on one image, [x_c, y_c, w, h]
  - calculate the jarccard index between the 8732 priors and N GT bounding box. Then we have two rules to match the priors to GT bounding box

  1. for each GT, find the best match priors

  2. for each prior, get the max IoU, if:

     - > 0.5, Positive match with the GT bounding box

     - <0.5, negative match with any GT bounding box

  - after matching, for each prior, we got **match type(positive or negative), target label, target coords.**
  - TPU can only access files in GCS, this is annoying because its impossible to read downloaded file to build a dataset when using TPU. We can only use tfds.
  - Finally,  we realized that the matching work should be done in the loss calculation or prepare stage, since the dataset is universe for different object detection methods and should not restrict to some specific methods

- transform:

  - parse the dict
  - resize
  - normalize by imagenet mean and std
  - data augmentation
  - match(anchors to gt)

- model:

  - base conv
  - aux conv
  - prediction conv

- Loss:

  - cls
  - loc

- defination:

  - example

    : an elem in the dataset, is a dict

    - form:
      - 'image' (original shape) uint8
      - 'objects' 'bbox' (N, 4) float32
      - 'objects' 'is_difficult' (N,) bool # not for now
      - 'objects' 'label' (N,) int64
      - other things
    - flow:
      - from dataset

  - anchor: a group of bboxes that is pre-defined in a detection method

    - form:
      - a list of dict, for each dict:
        - 'bbox'

  - iou (jaccard index)

    - cal:
      - its easy to implement in the bc form

  - bbox: a bounding box

    - form:
      - can be specified by bc or cc
    - type:
      - bbox of a anchor (8732): prior or default box
      - bbox of a pred (8732): bboxes the model proposed
      - bbox of a gt (N): the real ones
      - bbox of a match (8732): the real ones after matching

  - target:  the gt_bbox for each anchor_bbox

    - form:
      - a list of dict, each dict corresponds to a anchor, for each dict:
        - 'label': semi-positive value, 0 means bgd, otherwise coresponding label
        - 'offset': from the anchor to the gt_bbox
    - flow:
      - match anchors to gtbboxes
      - there are 2 rules:
        1. anchor wise
        2. gtbbox wise
      - should be done in the transform

  - pred(prediction)：

    - form:
      - a list of dict , for each dict (pred):
        - 'label': semi-positive value
        - 'pred_bbox'

  - SSDLoss: calculate the loss according to the SSD paper by

    - cls loss: positive + hard negative
    - loc loss:

  - coords: coordinates

    - form:
      - **bc** (boundary coordinates): in the form of (x_min, y_min, x_max, y_max)
      - **cc** (center-size coordinates): in the form of (x_c, y_c, w, h)
    - flow:
      - bc form is used in the dataset
      - in the transform, coords might be changed, in cc form at last
