from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
import re

class Noise_generator(object):
    '''
    Generator Arch.
    z = 128-d
    p = fully_connected(z)  - [batch_size, project_dim]
    h0 = reshape(p)         - [batch_size, 1, 32, 64]
    output = conv2D(h0)     - [batch_size, ngf*6, 32, 64]

    for i in range(3):
        output = residual_block(output)
        output = residual_block(output)
        output = residual_block(output)
        output = deconv2D(output) - [32, 64] -> [64, 64] -> [128, 64] -> [256, 64]

    final_output = conv2D(output)
    '''
    def __init__(self, n_shape, c_shape, name="noise_generator"):
        self.n_shape = n_shape
        self.c_shape = c_shape
        self.project_dim = 64*32
        self.initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        self.name = name
    def __call__(self, z, reuse=True):
        ngf = 16
        layer_specs = [
            ngf * 6,
            ngf * 12,
            ngf * 24,
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
                    h = tf.layers.batch_normalization(layers[-1], axis=1, name=name+'bn')
                    h = activation(h, 'prelu', name+'act')
                    output = deconv2D_up(h, out_channels*2, [12,3], [1,1,1,1], name=name)
                    layers.append(output)

                name = "output_layer"
                output = conv2d(layers[-1], 1, [12,3], [1,1,1,1], name=name)
                output = activation(output, 'tanh', name+'act')
                layers.append(output)

        print("Noise Generator:")                
        for l in layers:
            print(l.get_shape())
        return layers[-1]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Discriminator(object):
    '''
    Discriminator Arch.
    layers of conv2d with stride = [2,2] and layer_normalization
    flatten()
    fully_connected(256)
    fully_connected(1)
    '''
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
                if noisy != None:
                    input = tf.concat([noisy, clean], axis=1)
                else:
                    input = clean
                name = 'dlayer_1'
                convolved = conv2d(input, ndf, [3,3], [1,1,2,2], name=name)
                rectified = activation(convolved, 'lrelu', name+'_activation')
                layers.append(rectified)

                for i in range(n_layers):
                    name = "dlayer_%d" % (len(layers) + 1)
                    out_channels = ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = conv2d(layers[-1], out_channels, [3,3], [1,1,stride,stride], name=name)
                    normalized = layernorm(convolved, axis=[1, 2, 3], name=name+'_layernorm')
                    rectified = activation(normalized, 'lrelu', name+'_activation')
                    layers.append(rectified)

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
    '''
    Globe Discriminator Arch.
    layers of conv2d with stride = [2,2] and layer_normalization
    flatten()
    fully_connected(256)
    fully_connected(1)
    '''
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
                if noisy != None:
                    input = tf.concat([noisy, clean], axis=1)
                else:
                    input = clean
                name = 'dlayer_1'
                convolved = conv2d(input, ndf, [12,3], [1,1,2,2], name=name)
                rectified = activation(convolved, 'lrelu', name+'_activation')
                layers.append(rectified)

                for i in range(n_layers):
                    name = "dlayer_%d" % (len(layers) + 1)
                    out_channels = ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = conv2d(layers[-1], out_channels, [12,3], [1,1,stride,stride], name=name)
                    normalized = layernorm(convolved, axis=[1, 2, 3], name=name+'_layernorm')
                    rectified = activation(normalized, 'lrelu', name+'_activation')
                    layers.append(rectified)

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