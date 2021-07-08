import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Layer
from tensorflow.keras.applications import VGG16


# TODO: Add weight decay
# TODO: Add layer-wise learning rates
# TODO: separate the model with and without pretrained weights


class VGG_backbone_raw:
    def __init__(self, image_res=300):
        # pretrained layers
        self.vgg_layers = VGG16(include_top=True).layers

        # build the model
        _input = Input(shape=(image_res, image_res, 3))
        fm4_3, x = self.normal_layers(_input)
        fm4_3 = tf.math.l2_normalize(fm4_3, -1)  # l2 normalize to match the scale
        fm7 = self.fc_layers(x)
        self.model = Model(_input, [fm4_3, fm7])

        # load weights for fc-adapted layers
        self.load_weights(self.model)

        assert(self.model.layers[5].weights == self.vgg_layers[5].weights)

    def normal_layers(self, x):

        normal_layers = self.vgg_layers[:-4]
        normal_layers[10] = MaxPool2D((2, 2), (2, 2), 'same')
        normal_layers[18] = MaxPool2D((3, 3), (1, 1), 'same')

        for layer in normal_layers[1:14]:  # exclude the first Input layer
            x = layer(x)
        fm4_3 = x

        for layer in normal_layers[14:]:
            x = layer(x)

        return fm4_3, x

    def fc_layers(self, x):
        x = Conv2D(1024, (3, 3), (1, 1), 'same', dilation_rate=(6, 6),
                   activation='relu')(x)  # decimate (6, 6) since block5_pool does not half fm
        x = Conv2D(1024, (1, 1), (1, 1), 'same', activation='relu')(x)

        return x

    def load_weights(self, model):
        """only need to load weights for the fc-adapted layers,
        since the lower layers has weights inherently"""

        # fc1 adapted layer
        weights = self.vgg_layers[20].get_weights()
        weights[0] = weights[0].reshape([7, 7, 512, 4096])[::3, ::3, :, ::4]  # decimate the weights
        weights[1] = weights[1][::4]
        model.layers[-2].set_weights(weights)

        # fc2 adapted layer
        weights = self.vgg_layers[21].get_weights()
        weights[0] = weights[0].reshape([1, 1, 4096, 4096])[:, :, ::4, ::4]  # decimate the weights
        weights[1] = weights[1][::4]
        model.layers[-1].set_weights(weights)


def VGG_backbone(image_res=300):
    return VGG_backbone_raw(image_res).model


class SSDAuxConv(Layer):
    def __init__(self):
        super(SSDAuxConv, self).__init__()
        self.stages = [[
            Conv2D(256, (1, 1), activation='relu'),
            Conv2D(512, (3, 3), (2, 2), 'same', activation='relu')
        ], [
            Conv2D(128, (1, 1), activation='relu'),
            Conv2D(256, (3, 3), (2, 2), 'same', activation='relu')
        ], [
            Conv2D(128, (1, 1), activation='relu'),
            Conv2D(256, (3, 3), activation='relu')
        ], [
            Conv2D(128, (1, 1), activation='relu'),
            Conv2D(256, (3, 3), activation='relu')
        ]]

    def call(self, x, **kwargs):
        outputs = []
        for stage in self.stages:
            x = stage[0](x)
            x = stage[1](x)
            outputs.append(x)
        return outputs


class SSDPredConv(Layer):

    def __init__(self, priors_per_tile, fm_size):
        super(SSDPredConv, self).__init__()
        self.priors_per_tile = priors_per_tile
        self.fm_size = fm_size

        self.loc_conv = [Conv2D(priors * 4, (3, 3), padding='same')
                         for priors in self.priors_per_tile]
        self.cls_conv = [Conv2D(priors * 21, (3, 3), padding='same')
                         for priors in self.priors_per_tile]

    def call(self, fms, **kwargs):
        loc_output = []
        cls_output = []
        for i in range(len(fms)):
            loc_tmp = self.loc_conv[i](fms[i])  # H * W * 4
            loc_tmp = tf.reshape(loc_tmp, [-1, self.fm_size[i] * self.priors_per_tile[i], 4])  # reshape
            loc_output.append(loc_tmp)
            cls_tmp = self.cls_conv[i](fms[i])  # H * W * 21
            cls_tmp = tf.reshape(cls_tmp, [-1, self.fm_size[i] * self.priors_per_tile[i], 21])
            cls_tmp = tf.math.softmax(cls_tmp, axis=-1)  # softmax to get cls score
            cls_output.append(cls_tmp)
        return tf.concat(loc_output, 1), tf.concat(cls_output, 1)


class SSD(Model):
    def __init__(self,
                 backbone=VGG_backbone(300),
                 priors_per_tile=None,
                 fm_size=None):
        super(SSD, self).__init__()
        if priors_per_tile is None:
            priors_per_tile = [4, 6, 6, 6, 4, 4]
        if fm_size is None:
            fm_size = [38 * 38, 19 * 19, 10 * 10, 5 * 5, 3 * 3, 1 * 1]
        self.base_conv = backbone
        self.aux_conv = SSDAuxConv()
        self.pred_conv = SSDPredConv(priors_per_tile, fm_size)

    def call(self, x, **kwargs):
        fm4_3, fm7 = self.base_conv(x)
        fms = self.aux_conv(fm7)
        loc_output, cls_output = self.pred_conv([fm4_3, fm7]+fms)
        return {
            'offsets': loc_output,
            'classes': cls_output
        }
