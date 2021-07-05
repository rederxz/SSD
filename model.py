import math

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D
from tensorflow.keras.applications import VGG16


# TODO: Add weight decay
# TODO: Add layer-wise learning rates
# TODO: Backbone and other


class VGG16_backbone(Model):

    # build the whole model firstly and then get the wanted feats

    def __init__(self, image_res=300):
        super(VGG16_backbone, self).__init__()
        self.image_res = image_res
        self.vgg = VGG16(include_top=True)

        # the pretrained model

        # part1 - max_pool3 - part2 - max_pool5 - part3(from fc layers) - aux_conv - pred_conv

        # x = Input(shape=(image_res, image_res, 3))
        # # part1
        # x = Model(vgg16.input, vgg16.layers[9].output)(x)
        # # max_pool3
        # max_pool_3 = MaxPool2D((2, 2), (2, 2), 'same', name=vgg16.layer[10].name)
        # # part2
        # tmp_input = Input(shape=(image_res))

        layers = self.vgg.layers
        layers[10] = MaxPool2D((2, 2), (2, 2), 'same')
        layers[18] = MaxPool2D((3, 3), (1, 1), 'same')

        _input = Input(shape=(300, 300, 3))
        x = layers[1](_input)
        for layer in layers[2:-3]:
            x = layer(x)

        model = Model(_input, x)
        model.summary()

        print(model.layers[5].weights == self.vgg.layers[5].weights)

        # return model

    def call(self):
        pass

        # part_1 = Model(self.vgg.input, self.vgg.layers[9].output)
        #
        # max_pool_3 = MaxPool2D((2, 2), (2, 2), 'same')
        #
        # part_2 = Model(self.vgg.layers[11].input, self.vgg.layers[17].output)
        #
        # max_pool_5 = MaxPool2D((3, 3), (1, 1), 'same')
        #
        # part_3 = self.part_3()
        #
        # return Sequential([part_1, max_pool_3, part_2, max_pool_5, part_3])

    def part_1(self):
        _input = Input(shape=(self.image_res, self.image_res, 3))
        x = Model(self.vgg.input, self.vgg.layers[9].output)(_input)
        return Model(_input, x)

    def max_pool_3(self):
        return MaxPool2D((2, 2), (2, 2), 'same', name=self.vgg.layers[10].name)

    def part_2(self):
        res = math.ceil(self.image_res / 2)
        for i in range(2):
            res = math.ceil(res / 2)
        _input = Input(shape=(res, res, 256))
        x = Model(self.vgg.layer[11].input, self.vgg.layer[17].output)(_input)
        return Model(_input, x)

    def max_pool_5(self):
        return MaxPool2D((3, 3), (1, 1), 'same', name=self.layer[18].name)

    def part_3(self):
        conv_1 = Conv2D(1024, (3, 3), (1, 1), 'same',
                   dilation_rate=(6, 6),
                   activation=self.vgg.activation,
                   name=self.vgg.name)  # decimate (6, 6) because block5_pool do not downsample
        conv_2 = Conv2D(1024, (1, 1), (1, 1), 'same',
                   activation=self.vgg.activation,
                   name=self.vgg.name)
        return Sequential([conv_1, conv_2])



def SSD300():
    def base_conv(x):
        for i in range(1, 22):
            vgg_layer = vgg.get_layer(index=i)
            if 'conv' in vgg_layer.name:
                x = Conv2D(vgg_layer.filters,
                           vgg_layer.kernel_size,
                           vgg_layer.strides,
                           vgg_layer.padding,
                           activation=vgg_layer.activation,
                           name=vgg_layer.name)(x)
                if vgg_layer.name == 'block4_conv3':
                    fm4_3 = x
            elif 'pool' in vgg_layer.name:
                if '3' in vgg_layer.name:  # padding='same' to get 38x38
                    x = MaxPool2D((2, 2), (2, 2), 'same', name=vgg_layer.name)(x)
                elif '5' in vgg_layer.name:  # block5_pool from 2x2_2x2 to 3x3_1x1
                    x = MaxPool2D((3, 3), (1, 1), 'same', name=vgg_layer.name)(x)
                else:
                    x = MaxPool2D(vgg_layer.pool_size,
                                  vgg_layer.strides,
                                  vgg_layer.padding,
                                  # notion: in vggnet, all pooling layers' padding == 'valid' , is it ok?
                                  name=vgg_layer.name)(x)
            elif 'fc' in vgg_layer.name:
                if vgg_layer.name == 'fc1':
                    x = Conv2D(1024, (3, 3), (1, 1), 'same',
                               dilation_rate=(6, 6),
                               activation=vgg_layer.activation,
                               name=vgg_layer.name)(x)  # decimate (6, 6) because block5_pool do not downsample
                elif vgg_layer.name == 'fc2':
                    x = Conv2D(1024, (1, 1), (1, 1), 'same',
                               activation=vgg_layer.activation,
                               name=vgg_layer.name)(x)
                    fm7 = x

        return [fm4_3, fm7]

    def aux_conv(x):
        x = Conv2D(256, (1, 1), activation='relu')(x)
        fm8_2 = Conv2D(512, (3, 3), (2, 2), 'same', activation='relu', name='block8_conv2')(x)
        x = Conv2D(128, (1, 1), activation='relu')(fm8_2)
        fm9_2 = Conv2D(256, (3, 3), (2, 2), 'same', activation='relu', name='block9_conv2')(x)
        x = Conv2D(128, (1, 1), activation='relu')(fm9_2)
        fm10_2 = Conv2D(256, (3, 3), activation='relu', name='block10_conv2')(x)
        x = Conv2D(128, (1, 1), activation='relu')(fm10_2)
        fm11_2 = Conv2D(256, (3, 3), activation='relu', name='block11_conv2')(x)
        return [fm8_2, fm9_2, fm10_2, fm11_2]

    def pred_conv(x, priors_per_tile, fm_size):
        loc_output = []
        cls_output = []
        for i in range(len(fms)):
            loc_tmp = Conv2D(priors_per_tile[i] * 4, (3, 3), padding='same')(fms[i])  # H * W * 25
            loc_tmp = tf.reshape(loc_tmp, [-1, fm_size[i] * priors_per_tile[i], 4])  # reshape
            loc_output.append(loc_tmp)
            cls_tmp = Conv2D(priors_per_tile[i] * 21, (3, 3), padding='same')(fms[i])
            cls_tmp = tf.reshape(cls_tmp, [-1, fm_size[i] * priors_per_tile[i], 21])
            cls_tmp = tf.math.softmax(cls_tmp, axis=-1)  # softmax to get cls score
            cls_output.append(cls_tmp)
        loc_output = tf.concat(loc_output, 1)
        cls_output = tf.concat(cls_output, 1)

        return [loc_output, cls_output]

    def load_weight():
        # load weights from vggnet
        for i in range(1, 21):
            base_layer = model.get_layer(index=i)
            vgg_layer = vgg.get_layer(base_layer.name)
            if 'conv' in vgg_layer.name:
                base_layer.set_weights(vgg_layer.get_weights())
            elif 'fc' in vgg_layer.name:
                # print(vgg_layer.name)
                if vgg_layer.name == 'fc1':
                    weights = vgg_layer.get_weights()
                    weights[0] = weights[0].reshape([7, 7, 512, 4096])
                    weights[0] = weights[0][::3, ::3, :, ::4]  # decimate the weights
                    weights[1] = weights[1][::4]
                    base_layer.set_weights(weights)
                elif vgg_layer.name == 'fc2':
                    weights = vgg_layer.get_weights()
                    weights[0] = weights[0].reshape([1, 1, 4096, 4096])
                    weights[0] = weights[0][:, :, ::4, ::4]  # decimate the weights
                    weights[1] = weights[1][::4]
                    base_layer.set_weights(weights)

    vgg = VGG16()

    input_image = Input((300, 300, 3))

    # base conv
    fm4_3, fm7 = base_conv(input_image)

    # aux conv
    fms = aux_conv(fm7)

    # pred conv
    fms = [fm4_3, fm7] + fms
    priors_per_tile = [4, 6, 6, 6, 4, 4]
    fm_size = [38 * 38, 19 * 19, 10 * 10, 5 * 5, 3 * 3, 1 * 1]
    loc_output, cls_output = pred_conv(fms, priors_per_tile, fm_size)

    model = Model(input_image, {'output_offsets': loc_output, 'output_classes':cls_output})

    load_weight()

    return model


model = SSD300()
model.summary()

# model = VGG16_backbone()
# model.summary()
