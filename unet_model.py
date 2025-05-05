# -*- coding: utf-8 -*-
## Setup

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


def double_conv_block(x, n_filters):

    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def build_unet_model():
    NF = [64,128,256,512] # original
    #NF = [16,32,64,128]
    # inputs
    inputs = layers.Input(shape=(32,32,1))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, NF[0])
    # 2 - downsample
    f2, p2 = downsample_block(p1, NF[1])
    # 3 - downsample
    f3, p3 = downsample_block(p2, NF[2])
    # 4 - downsample
    f4, p4 = downsample_block(p3, NF[3])

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, NF[3]*2)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, NF[3])
    # 7 - upsample
    u7 = upsample_block(u6, f3, NF[2])
    # 8 - upsample
    u8 = upsample_block(u7, f2, NF[1])
    # 9 - upsample
    u9 = upsample_block(u8, f1, NF[0])

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss="binary_crossentropy",
                   metrics="accuracy")
    return unet_model

