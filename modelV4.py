import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine.training import _minimize

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.applications import VGG16


def load_SSDVGGConv_weights(model):
    # get vgg pretrained layers
    vgg_layers = VGG16(include_top=True).layers

    for i, layer in enumerate(vgg_layers):
        print(i)
        print(layer)

    # conv 1
    model.layers[0].set_weights(vgg_layers[1].get_weights() + vgg_layers[2].get_weights())
    # conv 2
    model.layers[2].set_weights(vgg_layers[4].get_weights() + vgg_layers[5].get_weights())
    # conv 3
    model.layers[4].set_weights(vgg_layers[7].get_weights() + vgg_layers[8].get_weights() +
                                vgg_layers[9].get_weights())
    # conv 4
    model.layers[6].set_weights(vgg_layers[11].get_weights() + vgg_layers[12].get_weights() +
                                vgg_layers[13].get_weights())
    # conv 5
    model.layers[8].set_weights(vgg_layers[15].get_weights() + vgg_layers[16].get_weights() +
                                vgg_layers[17].get_weights())
    # conv6 and conv7
    conv6_weights = vgg_layers[20].get_weights()
    conv6_weights[0] = conv6_weights[0].reshape([7, 7, 512, 4096])[::3, ::3, :, ::4]  # decimate the weights
    conv6_weights[1] = conv6_weights[1][::4]
    conv7_weights = vgg_layers[21].get_weights()
    conv7_weights[0] = conv7_weights[0].reshape([1, 1, 4096, 4096])[:, :, ::4, ::4]  # decimate the weights
    conv7_weights[1] = conv7_weights[1][::4]
    model.layers[10].set_weights(conv6_weights + conv7_weights)

    return model


class SSDSolver(Model):
    def train_step(self, data):
        """Because y is a dict, we need to each value respectively"""
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y['targets'], y_pred, sample_weight, regularization_losses=self.losses)

        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                  self.trainable_variables)

        self.compiled_metrics.update_state(y['gts'], y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Because y is a dict, we need to each value respectively"""
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        self.compiled_loss(
            y['targets'], y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y['gts'], y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


class SSDVGGConv(Model):
    def __init__(self):
        super(SSDVGGConv, self).__init__()
        self.conv_1 = Sequential([
            Conv2D(64, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(64, (3, 3), (1, 1), 'same', activation='relu'),
        ])
        self.pooling_1 = Sequential(
            [MaxPool2D((2, 2), (2, 2), 'same')]
        )
        self.conv_2 = Sequential([
            Conv2D(128, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(128, (3, 3), (1, 1), 'same', activation='relu'),
        ])
        self.pooling_2 = Sequential(
            [MaxPool2D((2, 2), (2, 2), 'same')]
        )
        self.conv_3 = Sequential([
            Conv2D(256, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(256, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(256, (3, 3), (1, 1), 'same', activation='relu'),
        ])
        self.pooling_3 = Sequential(
            [MaxPool2D((2, 2), (2, 2), 'same')]
        )
        self.conv_4 = Sequential([
            Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
        ])
        self.pooling_4 = Sequential(
            [MaxPool2D((2, 2), (2, 2), 'same')]
        )
        self.conv_5 = Sequential([
            Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
            Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
        ])
        self.pooling_5 = Sequential(
            [MaxPool2D((3, 3), (1, 1), 'same')]
        )
        self.conv_6_7 = Sequential([
            Conv2D(1024, (3, 3), (1, 1), 'same', dilation_rate=(6, 6), activation='relu'),
            Conv2D(1024, (1, 1), (1, 1), 'same', activation='relu')
        ])

    def call(self, x, **kwargs):
        x = self.conv_1(x)
        x = self.pooling_1(x)
        x = self.conv_2(x)
        x = self.pooling_2(x)
        x = self.conv_3(x)
        x = self.pooling_3(x)
        x = self.conv_4(x)
        # fm4_3 = tf.math.l2_normalize(x, -1)  # l2 normalize to match the scale
        fm4_3 = x
        x = self.pooling_4(fm4_3)
        x = self.conv_5(x)
        x = self.pooling_5(x)
        fm7 = self.conv_6_7(x)

        return [fm4_3, fm7]


class SSDAuxConv(Model):
    def __init__(self):
        super(SSDAuxConv, self).__init__()
        self.conv8 = Sequential([
            Conv2D(256, (1, 1), activation='relu'),
            Conv2D(512, (3, 3), (2, 2), 'same', activation='relu')
        ])
        self.conv9 = Sequential([
            Conv2D(128, (1, 1), activation='relu'),
            Conv2D(256, (3, 3), (2, 2), 'same', activation='relu')
        ])
        self.conv10 = Sequential([
            Conv2D(128, (1, 1), activation='relu'),
            Conv2D(256, (3, 3), activation='relu')
        ])
        self.conv11 = Sequential([
            Conv2D(128, (1, 1), activation='relu'),
            Conv2D(256, (3, 3), activation='relu')
        ])

    def call(self, x, **kwargs):
        outputs = []
        x = self.conv8(x)
        outputs.append(x)
        x = self.conv9(x)
        outputs.append(x)
        x = self.conv10(x)
        outputs.append(x)
        x = self.conv11(x)
        outputs.append(x)
        return outputs


class SSDPredConv(Model):
    priors_per_tile = [4, 6, 6, 6, 4, 4]

    def __init__(self):
        super(SSDPredConv, self).__init__()
        self.loc_conv = [Sequential([Conv2D(priors * 4, (3, 3), padding='same')]) for priors in self.priors_per_tile]
        self.cls_conv = [Sequential([Conv2D(priors * 21, (3, 3), padding='same')]) for priors in self.priors_per_tile]

    def call(self, fms, **kwargs):
        loc_output = []
        cls_output = []
        for i in range(len(fms)):
            fm_size = tf.reduce_prod(fms[i].shape[1:3])
            loc_tmp = self.loc_conv[i](fms[i])  # [N, H, W, 4 * priors]
            loc_tmp = tf.reshape(loc_tmp, [-1, fm_size * self.priors_per_tile[i], 4])  # reshape
            loc_output.append(loc_tmp)
            cls_tmp = self.cls_conv[i](fms[i])  # [N, H, W, 21 * priors]
            cls_tmp = tf.reshape(cls_tmp, [-1, fm_size * self.priors_per_tile[i], 21])
            cls_tmp = tf.math.softmax(cls_tmp, axis=-1)  # softmax to get cls score
            cls_output.append(cls_tmp)
        return tf.concat([tf.concat(loc_output, 1), tf.concat(cls_output, 1)], axis=-1)


class SSD(SSDSolver):
    def __init__(self):
        super(SSD, self).__init__()
        # priors_per_tile = [4, 6, 6, 6, 4, 4]
        # fm_size = [38 * 38, 19 * 19, 10 * 10, 5 * 5, 3 * 3, 1 * 1]
        self.base_conv = SSDVGGConv()
        self.aux_conv = SSDAuxConv()
        self.pred_conv = SSDPredConv()

    def call(self, x, **kwargs):
        fm_from_base_conv = self.base_conv(x)
        fm_from_aux_conv = self.aux_conv(fm_from_base_conv[-1])
        output = self.pred_conv(fm_from_base_conv + fm_from_aux_conv)
        return output


if __name__ == "__main__":
    # model = SSDVGGConv()
    # model.build((None, 300, 300, 3))
    # load_SSDVGGConv_weights(model)

    # model = SSDAuxConv()
    # model.build((None, 19, 19, 1024))

    # model = SSDPredConv()
    # model.build([(None, 38, 38, 16), (None, 5, 5, 64), (None, 5, 5, 64),
    #              (None, 5, 5, 64), (None, 5, 5, 64), (None, 5, 5, 64)])

    model = SSD()
    model.build((None, 300, 300, 3))
    model.call(tf.keras.Input((300, 300, 3)))
    model.summary()
