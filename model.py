from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
import re

class Noise_generator(object):
    def __init__(self, n_shape, c_shape, name="noise_generator"):
        self.n_shape = n_shape
        self.c_shape = c_shape
        self.project_dim = 64*32
        self.initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        self.name = name
    def __call__(self, z, reuse=True):
        ngf = 16
        layer_specs = [
            ngf * 6, # [batch, ngf*16, 32, 1] => [batch, ngf*16, 64, 1]
            ngf * 12, # [batch, ngf*16, 64, 1] => [batch, ngf*16, 128, 1]
            ngf * 24,  # [batch, ngf*16, 128, 1] => [batch, ngf*8, 256, 1]
        ]
        layers = []
        with tf.device('/gpu:0'):
            with tf.variable_scope(self.name) as vs:
                if reuse:
                    vs.reuse_variables()
                with tf.variable_scope("project_reshape"):                
                    p = fully_connected(z, self.project_dim, self.initializer)
                    p = activation(p, 'prelu', 'proj_act')
                    h0 = tf.reshape(p, [tf.shape(z)[0], 1, 32, 64])
                    h0 = conv2d(h0, layer_specs[0], [12,3], [1,1,1,1], name="conv1")
                    layers.append(h0)

                for layer_num, out_channels in enumerate(layer_specs):
                    name = "resblock_%d" % (layer_num)
                    output = residual_block(layers[-1], out_channels, [12,3], name=name+"_1")
                    layers.append(output)
                    output = residual_block(layers[-1], out_channels, [12,3], name=name+"_2")
                    layers.append(output)
                    output = residual_block(layers[-1], out_channels, [12,3], name=name+"_3")
                    layers.append(output)
                    # output = residual_block(layers[-1], out_channels, [12,3], name=name+"_4")
                    # layers.append(output)
                    # output = residual_block(layers[-1], out_channels, [12,3], name=name+"_5")
                    # layers.append(output)
                    # output = residual_block(layers[-1], out_channels, [12,3], name=name+"_6")
                    # layers.append(output)
                    h = tf.layers.batch_normalization(layers[-1], axis=1, name=name+'bn')
                    h = activation(h, 'prelu', name+'act')
                    output = deconv2D_up(h, out_channels*2, [12,3], [1,1,1,1], name=name)
                    layers.append(output)

                name = "output_layer"
                output = conv2d(layers[-1], 1, [12,3], [1,1,1,1], name=name)
                layers.append(output)

        print("Noise Generator:")                
        for l in layers:
            print(l.get_shape())
        return layers[-1]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Enhancer(object):
    def __init__(self, n_shape, c_shape, name="enhancer"):
        self.n_shape = n_shape
        self.c_shape = c_shape
        self.name = name
    def __call__(self, x, reuse=True):
        ''' Can be conditioned on `y` or not '''
        ngf = 16
        # nc, nh, nw = self.n_shape
        # cc, ch, cw = self.c_shape
        layers = []

        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            with tf.device('/gpu:0'):
                output = conv2d(x, ngf, [11,1], [1,1,1,1], name="encoder_1")
                layers.append(output)

                layer_specs = [
                    ngf * 1, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                    ngf * 2, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                    ngf * 2, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                    ngf * 4, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                    ngf * 4, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                    ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                    ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                    ngf * 16, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                    ngf * 16, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                    ngf * 32, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                ]

                for out_channels in layer_specs:
                    name = "encoder_%d" % (len(layers) + 1)
                    rectified = activation(layers[-1], 'prelu', name+'_activation')
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    output = conv2d(rectified, out_channels, [11,1], [1,1,2,1], name=name)
                    # output = batchnorm(convolved, axis=[1, 2, 3], name='G_layernorm')
                    layers.append(output)
            
            # ---------------------------------------------------------------------- #
            layer_specs = [
                (ngf * 32, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 16, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 16, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (ngf * 4, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 4, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (ngf * 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (ngf * 1, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            ]
            with tf.device('/gpu:0'):
                num_encoder_layers = len(layers)
                for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                    skip_layer = num_encoder_layers - decoder_layer - 1
                    name = "decoder_%d" % (skip_layer + 1)
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = layers[-1]
                    else:
                        input = tf.concat([layers[-1], layers[skip_layer]], axis=1)

                    rectified = activation(input, 'prelu', name+'_activation')
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    output = deconv_up(rectified, out_channels, [11,1], [1,1,1,1], name=name)
                    # output = batchnorm(output)

                    # if dropout > 0.0:
                    #     output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    layers.append(output)

                # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
                input = tf.concat([layers[-1], layers[0]], axis=1)
                name = 'decoder_1'
                rectified = activation(input, 'prelu', name+'_activation')
                output = conv2d(rectified, ngf, [11,1], [1,1,1,1], name=name)
                layers.append(output)

                name = 'output_layers'
                rectified = activation(layers[-1], 'prelu', name+'_activation')
                output = conv2d(rectified, 2, [3,1], [1,1,1,1], name=name)
                layers.append(output)

        print("Enhancer:")
        for l in layers:
            print(l.get_shape())

        return tf.split(layers[-1], 2, axis=1)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Discriminator(object):
    def __init__(self, name="discriminator"):
        self.name = name
    def __call__(self, noisy=None, clean=None, reuse=True):       
        n_layers = 3
        ndf = 16
        layers = []
        with tf.device('/gpu:0'):
            with tf.variable_scope(self.name) as vs:
                if reuse:
                    vs.reuse_variables()
                # 2x [batch, in_channels, height, width] => [batch, in_channels * 2, height, width]
                if noisy != None:
                    input = tf.concat([noisy, clean], axis=1)
                else:
                    input = clean
                name = 'dlayer_1'
                convolved = conv2d(input, ndf, [3,3], [1,1,2,2], name=name)
                rectified = activation(convolved, 'lrelu', name+'_activation')
                layers.append(rectified)

                # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
                # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
                # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
                for i in range(n_layers):
                    name = "dlayer_%d" % (len(layers) + 1)
                    out_channels = ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = conv2d(layers[-1], out_channels, [3,3], [1,1,stride,stride], name=name)
                    normalized = layernorm(convolved, axis=[1, 2, 3], name=name+'_layernorm')
                    rectified = activation(normalized, 'lrelu', name+'_activation')
                    layers.append(rectified)

                # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
                name = "dlayer_%d" % (len(layers) + 1)
                convolved = conv2d(layers[-1], 1, [3,3], [1,1,1,1], name=name)
                normalized = layernorm(convolved, axis=[1, 2, 3], name=name+'_layernorm')
                rectified = activation(normalized, 'lrelu', name+'_activation')
                layers.append(rectified)

                with tf.variable_scope("fully_connected"):
                    flatten = tf.contrib.layers.flatten(layers[-1])
                    fc1 = tf.contrib.layers.fully_connected(flatten, 256, activation_fn=None)
                    normalized = layernorm(fc1, axis=[1], name='layernorm')
                    rectified = activation(normalized, 'lrelu')
                    final = tf.contrib.layers.fully_connected(rectified, 1, activation_fn=None)
                    layers.append(final)

        print("Discriminator:")
        for l in layers:
            print(l.get_shape())

        return layers[-1]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name] 

class glob_Discriminator(object):
    def __init__(self, name="discriminator"):
        self.name = name
    def __call__(self, noisy=None, clean=None, reuse=True):       
        n_layers = 4
        ndf = 32
        layers = []
        with tf.device('/gpu:0'):
            with tf.variable_scope(self.name) as vs:
                if reuse:
                    vs.reuse_variables()
                # 2x [batch, in_channels, height, width] => [batch, in_channels * 2, height, width]
                if noisy != None:
                    input = tf.concat([noisy, clean], axis=1)
                else:
                    input = clean
                name = 'dlayer_1'
                convolved = conv2d(input, ndf, [12,3], [1,1,2,2], name=name)
                rectified = activation(convolved, 'lrelu', name+'_activation')
                layers.append(rectified)

                # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
                # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
                # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
                for i in range(n_layers):
                    name = "dlayer_%d" % (len(layers) + 1)
                    out_channels = ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = conv2d(layers[-1], out_channels, [12,3], [1,1,stride,stride], name=name)
                    normalized = layernorm(convolved, axis=[1, 2, 3], name=name+'_layernorm')
                    rectified = activation(normalized, 'lrelu', name+'_activation')
                    layers.append(rectified)

                # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
                name = "dlayer_%d" % (len(layers) + 1)
                convolved = conv2d(layers[-1], 1, [3,3], [1,1,1,1], name=name)
                normalized = layernorm(convolved, axis=[1, 2, 3], name=name+'_layernorm')
                rectified = activation(normalized, 'lrelu', name+'_activation')
                layers.append(rectified)

                with tf.variable_scope("fully_connected"):
                    flatten = tf.contrib.layers.flatten(layers[-1])
                    fc1 = tf.contrib.layers.fully_connected(flatten, 256, activation_fn=None)
                    normalized = layernorm(fc1, axis=[1], name='layernorm')
                    rectified = activation(normalized, 'lrelu')
                    final = tf.contrib.layers.fully_connected(rectified, 1, activation_fn=None)
                    layers.append(final)

        print("Discriminator:")
        for l in layers:
            print(l.get_shape())

        return layers[-1]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name] 